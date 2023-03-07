#include "SGR_Recorder.hpp"
#include "milkDebugTools.h"
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <limits>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

#include "SGR_ImageHandler.hpp"
#include "../util/SpotFitter.hpp"

extern "C"
{

struct data {
    size_t n;
    double * y;
    double * sigma;
};

SGR_Recorder::SGR_Recorder(
    IMAGE* in,
    IMAGE* dark,
    float pxSize,
    float mlaPitch,
    float mlaDist,
    const char* streamPrefix)
    :
    mpInput(in),
    mpDark(dark),
    mPxSize(pxSize),
    mMlaPitch(mlaPitch),
    mMlaDist(mlaDist),
    mStreamPrefix(streamPrefix)
{
    mApertureDiameter = mlaPitch/mPxSize;
    mGridRectSize = floor(mApertureDiameter);

    // Check if input and dark images are compatible
    if (mpInput->md->naxis != 2 || mpDark->md->naxis != 2)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Input- and dark image have to be two-dimensional.";
    }
    else
    {
        // Set up the dark subtraction image handler
        mIHdarkSubtract = newImHandlerFrmIm(float, makeStreamname("darksub"), mpInput);
        mIHdarkSubtract->cpy_subtract(mpInput, mpDark);
        mImgWidth = mIHdarkSubtract->mWidth;
        mImgHeight = mIHdarkSubtract->mHeight;
        mNumPixels = mIHdarkSubtract->mNumPx;
    }
}

errno_t SGR_Recorder::sampleDo()
{
    // Collect image statistics for the thresholding
    double avg = mIHdarkSubtract->getSumOverROI()/mIHdarkSubtract->getPxInROI();
    float maxVal = mIHdarkSubtract->getMaxInROI();
    float thresh = avg + (maxVal-avg)/4.;
    printf("Image statistics: Max=%.3f, AVG=%.3f => Thresh=%.3f\n",
        maxVal, avg, thresh);

    // Apply thresholding according to the statistics
    mIHthresh = newImHandlerFrmIm(uint8_t, makeStreamname("thresh"), mpInput);
    mIHthresh->cpy_thresh(mIHdarkSubtract->getImage(), thresh);

    // Get spot center estimates by erosion of the thresholded image
    mIHerode = newImHandlerFrmIm(uint8_t, makeStreamname("erode"), mIHthresh->getImage());
    std::vector<Point<uint32_t>> particles;
    while (mIHerode->erode(&particles) > 0);
    printf("Number of particles after thresholding: %d\n",
        (int) particles.size());
    
    // Filter the points by distance
    // Points that have neighbours which are too close are likely false positives.
    // The threshold is determined by the subaperture diameter
    // All particles that are left are used for the spot size measurement.
    std::vector<Point<uint32_t>> particlesFiltered =
                filterByMinDistance(particles, 0.8*mApertureDiameter);
    
    // Generate a model of the spots by fitting the selected particles
    SpotFitter spotFitter(mIHdarkSubtract);
    spotFitter.expressFit(
        particlesFiltered, mGridRectSize, 2, mVisualize);
    
    // Span a preliminary search grid, based on the fit positions
    std::vector<Point<double>> fitSpots = spotFitter.getFittedSpotCenters();
    spanCoarseGrid(spotFitter.getFittedSpotCenters());

    // Generate a gaussian convolution kernel which matches the stdDev
    // of the spots in the image
    double avgStdDev = spotFitter.getAvgStdDev();
    printf("Average standard deviation of spot PSF: %.3f\n", avgStdDev);

    

    
    

// =====================================

    switch(mState) {
    case RECSTATE::ERROR: return RETURN_FAILURE;
    case RECSTATE::INIT: mState = RECSTATE::RECORD; return RETURN_SUCCESS;
    case RECSTATE::RECORD:  mState = RECSTATE::FINISH; return RETURN_SUCCESS;
    case RECSTATE::FINISH:
    default: return RETURN_SUCCESS;
    }
}

const char* SGR_Recorder::getStateDescription()
{
    switch(mState) {
    case RECSTATE::ERROR: return makeErrorMessage().c_str();
    case RECSTATE::INIT: return "Initializing...\n";
    case RECSTATE::RECORD: return "Recording...\n";
    case RECSTATE::FINISH: return "Done!\n";
    default: return "UNKNOWN\n";
    }
}

std::string SGR_Recorder::makeErrorMessage()
{
    std::string msg("Error! Error message: \n\t");
    msg.append(mErrDescr);
    msg.append("\n");
    return msg;
}

const char* SGR_Recorder::makeStreamname(const char* name)
{
    std::string streamName = mStreamPrefix;
    streamName.append(name);
    return streamName.c_str();
}

std::vector<Point<uint32_t>> SGR_Recorder::filterByMinDistance(
    std::vector<Point<uint32_t>> pIn,
    double minDistance)
{
    std::vector<Point<uint32_t>> pFiltered;
    float keepoutDistance = 0.8*mApertureDiameter;
    for (int i = 0; i < pIn.size(); i++)
    {
        double minDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < pIn.size(); j++)
            if (i != j)
            {
                double distance = pIn.at(i).distance(pIn.at(j));
                if (distance < minDistance)
                    minDistance = distance;
                if (minDistance < keepoutDistance)
                    break;
            }
        if (minDistance >= keepoutDistance)
            pFiltered.push_back(pIn.at(i));
    }
    printf("Number of spots after filtering: %d\n",
        (int) pFiltered.size());
    
    return pFiltered;
}

void SGR_Recorder::spanCoarseGrid(std::vector<Point<double>> fitSpots)
{
    Point<double> gridAnchor(0, 0);
    for (size_t i = 0; i < fitSpots.size(); i++)
        gridAnchor += fitSpots.at(i) % mApertureDiameter;
    gridAnchor = (gridAnchor / fitSpots.size()) + mApertureDiameter/2;
    gridAnchor = gridAnchor % mApertureDiameter;
    mGridSize.mX = floor((mImgWidth - gridAnchor.mX)/mApertureDiameter);
    mGridSize.mY = floor((mImgHeight - gridAnchor.mY)/mApertureDiameter);
    printf("Grid anchor: X=%.3f, Y=%.3f\n", gridAnchor.mX, gridAnchor.mY);
    printf("Grid size: X=%d, Y=%d\n", mGridSize.mX, mGridSize.mY);
    printf("Grid rectangle size: %d\n", mGridRectSize);

    mGrid.clear();
    for (size_t ix = 0; ix < mGridSize.mX; ix++)
    {
        std::vector<Rectangle<uint32_t>> gridRow;
        for (size_t iy = 0; iy < mGridSize.mY; iy++)
        {
            uint32_t rootX = round(gridAnchor.mX + ix * mApertureDiameter);
            uint32_t rootY = round(gridAnchor.mY + iy * mApertureDiameter);
            Rectangle<uint32_t> r(rootX, rootY, mGridRectSize, mGridRectSize);
            gridRow.push_back(r);
        }
        mGrid.push_back(gridRow);
    }

    if (mVisualize)
    {
        mIHgridVisualization = newImHandlerFrmIm(float,
            makeStreamname("gridShow"), mIHdarkSubtract->getImage());
        // Copy the SHS image as visual reference
        mIHgridVisualization->cpy(mIHdarkSubtract->getImage());
        float visMax = mIHgridVisualization->getMaxInROI();
        // Paint the outlines of the grid
        for (size_t ix = 0; ix < mGridSize.mX; ix++)
            for (size_t iy = 0; iy < mGridSize.mY; iy++)
            {
                Rectangle<uint32_t> serachRect = mGrid.at(ix).at(iy);
                size_t rootX = serachRect.mRoot.mX;
                size_t rootY = serachRect.mRoot.mY;
                uint32_t upperEdge = rootY+mGridRectSize-1;
                uint32_t rightEdge = rootX+mGridRectSize-1;
                for (size_t h = 0; h < mGridRectSize; h++)
                {
                    mIHgridVisualization->write(visMax, rootX+h, rootY);
                    mIHgridVisualization->write(visMax, rootX+h, upperEdge);
                }
                for (size_t v = 1; v < mGridRectSize - 1; v++)
                {
                    mIHgridVisualization->write(visMax, rootX, rootY+v);
                    mIHgridVisualization->write(visMax, rightEdge, rootY+v);
                }
            }
        mIHgridVisualization->updateWrittenImage();
    }
}

}