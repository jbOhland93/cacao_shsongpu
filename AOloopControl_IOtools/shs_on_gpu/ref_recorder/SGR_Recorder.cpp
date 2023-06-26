#include "SGR_Recorder.hpp"
#include "milkDebugTools.h"

#include "SGR_ImageHandler.hpp"
#include "../util/SpotFitter.hpp"

#include <string>

extern "C"
{

SGR_Recorder::SGR_Recorder(
    IMAGE* in,
    IMAGE* dark,
    float pxSize_um,
    float mlaPitch_um,
    float mlaDist_um,
    uint32_t numSamples,
    const char* streamPrefix,
    bool visualize)
    :
    mpInput(in),
    mpDark(dark),
    mPxSize_um(pxSize_um),
    mMlaPitch_um(mlaPitch_um),
    mMlaDist_um(mlaDist_um),
    mSamplesExpected(numSamples),
    mStreamPrefix(streamPrefix),
    mVisualize(visualize)
{
    // Check if input and dark images are compatible
    if (mpInput->md->naxis != 2 || mpDark->md->naxis != 2)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Input- and dark image have to be two-dimensional.";
    }
    else
    {
        try
        {
            // Prepare the test stream prefix
            mTeststreamPrefix = mStreamPrefix;
            mTeststreamPrefix.append("Vrfy-");
            // Initialize parameters
            mApertureDiameter_px = mlaPitch_um/mPxSize_um;
            mGridRectSize = floor(mApertureDiameter_px);

            // Prepare the actual spotfinding
            prepareSpotFinding();
            mState = RECSTATE::AWAIT_SAMPLE;
        }
        catch (std::runtime_error e)
        {
            mState = RECSTATE::ERROR;
            mErrDescr = "Initialization error:";
            mErrDescr.append(e.what());
        }
    }
}

errno_t SGR_Recorder::sampleDo()
{
    if (mState != RECSTATE::AWAIT_SAMPLE)
    {
        mErrDescr = "Error while sampling: ";
        mErrDescr.append("State != AWAIT_SAMPLE, cannot evaluate sample.\n");
        mErrDescr.append("\tCurrent state: ");
        mErrDescr.append(getStateDescription());
        return RETURN_FAILURE;
    }
    
    mState = RECSTATE::SAMPLING;

    try
    {
        // Do dark subtraction
        mIHdarkSubtract->cpy_subtract(mpInput, mpDark);
        // Convolve the dark-subtracted image with the gaussian kernel
        mIHconvolution->cpy_convolve(
            mIHdarkSubtract->getImage(), mIHkernel->getImage());

        // Do Spotfinding with subpixel precision
        for (int gX = 0; gX < mGridSize.mX; gX++)
            for (int gY = 0; gY < mGridSize.mY; gY++)
            {
                Rectangle<uint32_t> searchRect = mGrid.at(gX).at(gY);

                // Move to subaperture ROI to find inter peak
                mIHconvolution->setROI(searchRect);
                uint32_t mpX;
                uint32_t mpY;
                float I_xy = mIHconvolution->getMaxInROI(&mpX, &mpY);
                
                // Go to absolute ROI again
                mIHconvolution->unsetROI();
                mpX += searchRect.mRoot.mX;
                mpY += searchRect.mRoot.mY;

                // Read intensites around the integer peak
                float I_xP_y = mIHconvolution->read(mpX+1,mpY);
                float I_xM_y = mIHconvolution->read(mpX-1,mpY);
                float I_x_yP = mIHconvolution->read(mpX,mpY+1);
                float I_x_yM = mIHconvolution->read(mpX,mpY-1);
                // Calculate sub-pixel shift (Poyneer 2003)
                float Dx = 0.5 * (I_xM_y - I_xP_y) / (I_xM_y + I_xP_y - 2*I_xy);
                float Dy = 0.5 * (I_x_yM - I_x_yP) / (I_x_yM + I_x_yP - 2*I_xy);

                // Record results
                mIHintensityREC->write(I_xy, gX, gY);
                mIHposREC->write(mpX + Dx, gX, gY);
                mIHposREC->write(mpY + Dy, gX + mGridSize.mX, gY);
            }

        mIHintensityREC->updateWrittenImage();
        mIHposREC->updateWrittenImage();

        mSamplesAdded++;
        if (mSamplesAdded < mSamplesExpected)
            mState = RECSTATE::AWAIT_SAMPLE;
        else
        {
            // Check if the recording buffers are completely filled once
            bool intFullCycle = mIHintensityREC->getImage()->md->CBcycle == 1;
            bool posFullCycle = mIHposREC->getImage()->md->CBcycle == 1;
            // Check if the recording buffers are exactly filled once
            bool intIntCycle = mIHintensityREC->getImage()->md->CBindex == 0;
            bool posIntCycle = mIHposREC->getImage()->md->CBindex == 0;

            if (intFullCycle && posFullCycle && intIntCycle && posIntCycle)
            {
                mState = RECSTATE::READY_FOR_EVAL;
                return RETURN_SUCCESS;
            }
            else
            {
                mState = RECSTATE::ERROR;
                mErrDescr = "The recording buffers are not filled exactly once!";
                return RETURN_FAILURE;
            }
        }

        return RETURN_SUCCESS;
    }
    catch(const std::exception& e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error while sampling: ";
        mErrDescr.append(e.what());
        return RETURN_FAILURE;
    }
}

errno_t SGR_Recorder::evaluateRecBuffers(float uradPrecisionThresh)
{
    if (mState != RECSTATE::READY_FOR_EVAL)
    {
        mErrDescr = "Error attempting to start evaluation: ";
        mErrDescr.append("State != READY_FOR_EVAL, cannot start evaluation.\n");
        mErrDescr.append("\tCurrent state: ");
        mErrDescr.append(getStateDescription());
        return RETURN_FAILURE;
    }

    mState = RECSTATE::EVALUATING;
    printf("\n\n=== SGR_Recorder: Evaluating recorded buffers ===\n\n");

    // Check if the recording buffers are completely filled once
    bool intFullCycle = mIHintensityREC->getImage()->md->CBcycle == 1;
    bool posFullCycle = mIHposREC->getImage()->md->CBcycle == 1;
    // Check if the recording buffers are exactly filled once
    bool intIntCycle = mIHintensityREC->getImage()->md->CBindex == 0;
    bool posIntCycle = mIHposREC->getImage()->md->CBindex == 0;

    bool cbOK = intFullCycle && posFullCycle && intIntCycle && posIntCycle;
    if (!cbOK)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error during evaluation: ";
        mErrDescr.append("The recording buffers are not filled exactly once!");
        return RETURN_FAILURE;
    }
    
    try {
        printf("Trying\n");
        // Initialize the  image handlers for the evaluation
        spImageHandler(float) IHavgI = SGR_ImageHandler<float>::newImageHandler(
            makeStreamname("7_Eval-AVGintensity"), mGridSize.mX, mGridSize.mY);
        IHavgI->setPersistent(mVisualize);
        spImageHandler(float) IHavgP = SGR_ImageHandler<float>::newImageHandler(
            makeStreamname("7_Eval-AVGpos"), mGridSize.mX*2, mGridSize.mY);
        IHavgP->setPersistent(mVisualize);
        spImageHandler(float) IHstdDvP = SGR_ImageHandler<float>::newImageHandler(
            makeStreamname("7_Eval-STDDVpos"), mGridSize.mX*2, mGridSize.mY);
        IHstdDvP->setPersistent(mVisualize);
        spImageHandler(uint8_t) IHspotMask = SGR_ImageHandler<uint8_t>::newImageHandler(
            makeStreamname("7_Eval-SpotMask"), mGridSize.mX, mGridSize.mY);
        IHspotMask->setPersistent(true);

        // Calculate the stability threshold for the mask
        float deflectionPrecision = mMlaDist_um * tan(uradPrecisionThresh*1e-6);
        float pxPrecision = deflectionPrecision / mPxSize_um;
        printf("Demanded spot stability: %.1f µrad == %.4f px. Generating mask...\n",
            uradPrecisionThresh, pxPrecision);
        
        // Size of the circular buffer == # of frames recorded
        uint32_t cbSize = mIHintensityREC->getImage()->md->CBsize;
        // Initialize fields for statistics
        std::vector<float> vI, vX, vY;
        float avgI, avgX, avgY, stdDvX, stdDvY, stdDvXY;

        int numOfValidSpots = 0;
        
        for (uint32_t ix = 0; ix < mGridSize.mX; ix++)
            for (uint32_t iy = 0; iy < mGridSize.mY; iy++)
            {
                vI = mIHintensityREC->readCircularBufAt(ix,iy);
                vX = mIHposREC->readCircularBufAt(ix,iy);

                vY = mIHposREC->readCircularBufAt(ix+mGridSize.mX,iy);
                avgI = avgX = avgY = 0.;
                for (uint32_t iz = 0; iz < cbSize; iz++)
                {
                    avgI += vI.at(iz);
                    avgX += vX.at(iz);
                    avgY += vY.at(iz);
                }
                avgI /= cbSize;
                avgX /= cbSize;
                avgY /= cbSize;
                IHavgI->write(avgI, ix, iy);
                IHavgP->write(avgX, ix, iy);
                IHavgP->write(avgY, ix+mGridSize.mX, iy);

                stdDvX = stdDvY = stdDvXY = 0.;
                for (uint32_t iz = 0; iz < cbSize; iz++)
                {
                    stdDvX += pow(vX.at(iz)-avgX, 2);
                    stdDvY += pow(vY.at(iz)-avgY, 2);
                }
                stdDvXY = sqrt((stdDvX+stdDvY)/cbSize);
                stdDvX = sqrt(stdDvX/cbSize);
                stdDvY = sqrt(stdDvY/cbSize);

                IHstdDvP->write(stdDvX, ix, iy);
                IHstdDvP->write(stdDvY, ix+mGridSize.mX, iy);
                if (stdDvXY <= pxPrecision)
                {
                    IHspotMask->write(1, ix, iy);
                    numOfValidSpots++;
                }
                else
                    IHspotMask->write(0, ix, iy);
            }

        IHavgI->updateWrittenImage();
        IHavgP->updateWrittenImage();
        IHstdDvP->updateWrittenImage();
        IHspotMask->updateWrittenImage();
        printf("Mask generated. %d valid subapertures detected.\n", numOfValidSpots);
        printf("\nTODOs: Generate 1D arrays of centroids, search rects, and the kernel, load them to GPU.\n\n");

        // Make GPU reference

        std::string refName = makeStreamname(mpInput->name);
        std::string maskName = refName;
        refName.append("_Ref");
        maskName.append("_Mask");
        

        IMAGE gpuRef;
        ImageStreamIO_createIm_gpu(
            &gpuRef,
            refName.c_str(),
            mpInput->md->naxis,
            mpInput->md->size,
            _DATATYPE_FLOAT,
            mDevice,            // -1: CPU RAM, 0+ : GPU
            1,                  // shared?
            0,                  // # of semaphores
            5,                  // # of keywords
            mpDark->md->imagetype,
            0 // circular buffer size (if shared), 0 if not used
        );
        // Make 1D arrays
        // Store fits files
        // Generate image streams on GPU
        // Load 1D arrays onto GPU

        

        mState = RECSTATE::FINISH;

        return RETURN_SUCCESS;
    } catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error during evaluation: ";
        mErrDescr.append(e.what());
        return RETURN_FAILURE;
    }
}

const char* SGR_Recorder::getStateDescription()
{

    switch(mState) {
    case RECSTATE::ERROR:
        mStateDescr = "SGR_Recorder Error: \n\t";
        mStateDescr.append(mErrDescr);
        mStateDescr.append("\n");
        break;
    case RECSTATE::INITIALIZING:
        mStateDescr = "Initializing...\n";
        break;
    case RECSTATE::AWAIT_SAMPLE:
        mStateDescr = "Collected ";
        mStateDescr.append(std::to_string(mSamplesAdded));
        mStateDescr.append("/");
        mStateDescr.append(std::to_string(mSamplesExpected));
        mStateDescr.append(" samples. Awaiting next sample ...\n");
        break;
    case RECSTATE::SAMPLING:
        mStateDescr = "Evaluating sample...\n";
        break;
    case RECSTATE::READY_FOR_EVAL: 
        mStateDescr = "Collected ";
        mStateDescr.append(std::to_string(mSamplesAdded));
        mStateDescr.append("/");
        mStateDescr.append(std::to_string(mSamplesExpected));
        mStateDescr.append(". Ready for evaluation...\n");
        break;
    case RECSTATE::EVALUATING: 
        mStateDescr = "Evaluating recorded buffers ...\n";
        break;
    case RECSTATE::FINISH:
        mStateDescr = "Done!\n";
        break;
    default: mStateDescr = "UNKNOWN\n";
    }

    return mStateDescr.c_str();
}

const char* SGR_Recorder::makeStreamname(const char* name)
{
    mTmpStreamName = mStreamPrefix;
    mTmpStreamName.append(name);
    return mTmpStreamName.c_str();
}

const char* SGR_Recorder::makeTestStreamname(const char* name)
{
    mTmpStreamName = mTeststreamPrefix;
    mTmpStreamName.append(name);
    return mTmpStreamName.c_str();
}

void SGR_Recorder::prepareSpotFinding()
{
    if (mState != RECSTATE::INITIALIZING)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "SGR_Recorder::prepareSpotFinding: Already initialized.\n";
        return;
    }

    printf("\n\n=== SGR_Recorder: Preparing spot finding ===\n\n");
    // Set up the dark subtraction image handler
    try {
        mIHdarkSubtract = newImHandlerFrmIm(float, makeStreamname("1-darksub"), mpInput);
        mIHdarkSubtract->setPersistent(mVisualize);
        mImgWidth = mIHdarkSubtract->mWidth;
        mImgHeight = mIHdarkSubtract->mHeight;
        mNumPixels = mIHdarkSubtract->mNumPx;
        // Do dark subtraction
        mIHdarkSubtract->cpy_subtract(mpInput, mpDark);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error while setting up dark subtraction: ";
        mErrDescr.append(e.what());
        return;
    }

    // Collect image statistics for the thresholding
    float thresh;
    try {
        double avg = mIHdarkSubtract->getSumOverROI()/mIHdarkSubtract->getPxInROI();
        float maxVal = mIHdarkSubtract->getMaxInROI();
        thresh = avg + (maxVal-avg)/4.;
        printf("Image statistics: Max=%.3f, AVG=%.3f => Thresh=%.3f\n",
            maxVal, avg, thresh);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error while collecting image statistics: ";
        mErrDescr.append(e.what());
        return;
    }

    // Apply thresholding according to the statistics
    try {
        mIHthresh = newImHandlerFrmIm(uint8_t,
            makeStreamname("2-thresh"), mpInput);
        mIHthresh->setPersistent(mVisualize);
        mIHthresh->cpy_thresh(mIHdarkSubtract->getImage(), thresh);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error while thresholding: ";
        mErrDescr.append(e.what());
        return;
    }

    // Get spot center estimates by erosion of the thresholded image
    std::vector<Point<uint32_t>> particlesFiltered;
    try {
        mIHerode = newImHandlerFrmIm(uint8_t,
            makeStreamname("3-erode"), mIHthresh->getImage());
        mIHerode->setPersistent(mVisualize);
        std::vector<Point<uint32_t>> particles;
        while (mIHerode->erode(&particles) > 0);
        printf("Number of particles after thresholding: %d\n",
            (int) particles.size());
        
        // Filter the points by distance
        // Points that have neighbours which are too close are likely false positives.
        // The threshold is determined by the subaperture diameter
        // All particles that are left are used for the spot size measurement.
        particlesFiltered =
                    filterByMinDistance(particles, 0.8*mApertureDiameter_px);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error while selecting particles via erosion: ";
        mErrDescr.append(e.what());
        return;
    }
    
    // Generate a model of the spots by fitting the selected particles
    try {
        SpotFitter spotFitter(mIHdarkSubtract);
        spotFitter.expressFit(
            particlesFiltered, mGridRectSize, 2,
            mTeststreamPrefix, mVisualize);    
    
        // Span a preliminary search grid, based on the fit positions
        std::vector<Point<double>> fitSpots = spotFitter.getFittedSpotCenters();
        spanCoarseGrid(spotFitter.getFittedSpotCenters());

        // Generate a gaussian convolution kernel which matches the stdDev
        // of the spots in the image
        double stdDeviation = spotFitter.getAvgStdDev();
        printf("Average standard deviation of spot PSF: %.3f\n", stdDeviation);
        buildKernel(stdDeviation);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error during spot fitting and fit evaluation: ";
        mErrDescr.append(e.what());
        return;
    }
    
    try {
        // Prepare the convolution image stream
        mIHconvolution = newImHandlerFrmIm(float,
            makeStreamname("5-convol"), mpInput);
        mIHconvolution->setPersistent(mVisualize);

        // Prepare the averaging image streams for intensity and positions
        mIHintensityREC = SGR_ImageHandler<float>::newImageHandler(
                makeStreamname("6-recordAmp"),
                mGridSize.mX, mGridSize.mY,
                0,
                mSamplesExpected);
        mIHintensityREC->setPersistent(mVisualize);
        mIHposREC = SGR_ImageHandler<float>::newImageHandler(
                makeStreamname("6-RecordSpotPos"),
                mGridSize.mX*2,
                mGridSize.mY,
                0,
                mSamplesExpected);
        mIHposREC->setPersistent(mVisualize);
    }
    catch (std::runtime_error e)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Error during stream preparation: ";
        mErrDescr.append(e.what());
        return;
    }

    printf("\n\n=== SGR_Recorder: Preparation complete. Ready to collect samples. ===\n");
}

std::vector<Point<uint32_t>> SGR_Recorder::filterByMinDistance(
    std::vector<Point<uint32_t>> pIn,
    double minDistance)
{
    std::vector<Point<uint32_t>> pFiltered;
    float keepoutDistance = 0.8*mApertureDiameter_px;
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
        gridAnchor += fitSpots.at(i) % mApertureDiameter_px;
    gridAnchor = (gridAnchor / fitSpots.size()) + mApertureDiameter_px/2;
    gridAnchor = gridAnchor % mApertureDiameter_px;
    mGridSize.mX = floor((mImgWidth - gridAnchor.mX)/mApertureDiameter_px);
    mGridSize.mY = floor((mImgHeight - gridAnchor.mY)/mApertureDiameter_px);
    printf("Grid anchor: X=%.3f, Y=%.3f\n", gridAnchor.mX, gridAnchor.mY);
    printf("Grid size: X=%d, Y=%d\n", mGridSize.mX, mGridSize.mY);
    printf("Grid rectangle size: %d\n", mGridRectSize);

    mGrid.clear();
    for (size_t ix = 0; ix < mGridSize.mX; ix++)
    {
        std::vector<Rectangle<uint32_t>> gridRow;
        for (size_t iy = 0; iy < mGridSize.mY; iy++)
        {
            uint32_t rootX = round(gridAnchor.mX + ix * mApertureDiameter_px);
            uint32_t rootY = round(gridAnchor.mY + iy * mApertureDiameter_px);
            Rectangle<uint32_t> r(rootX, rootY, mGridRectSize, mGridRectSize);
            gridRow.push_back(r);
        }
        mGrid.push_back(gridRow);
    }

    if (mVisualize)
    {
        mIHgridVisualization = newImHandlerFrmIm(float,
            makeTestStreamname("gridShow"), mIHdarkSubtract->getImage());
        mIHgridVisualization->setPersistent(true);
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

void SGR_Recorder::buildKernel(double stdDev)
{
    // Determine the kernel size
    // => Include 4 stdDevs
    uint32_t kernelSize = ceil(4*stdDev);
    if ((kernelSize % 2) == 0)
        kernelSize ++;  // The kernel size should be odd
    float kernelCenter = floor(kernelSize/2);
    printf("Kernel size = %d, kernel center @ %.0f\n", kernelSize, kernelCenter);
    
    // Generate the kernel image handler
    mIHkernel = SGR_ImageHandler<float>::newImageHandler(
        makeStreamname("4-kernel"), kernelSize, kernelSize);
    mIHkernel->setPersistent(mVisualize);
    
    // Build the kernel
    float kernelSum = 0;
    for (uint32_t ix = 0; ix < kernelSize; ix++)
        for (uint32_t iy = 0; iy < kernelSize; iy++)
        {
            float x = ix-kernelCenter;
            float y = iy-kernelCenter;
            float val = exp(-(x*x+y*y)/(2*stdDev*stdDev));
            mIHkernel->write(val, ix, iy);
            kernelSum += val;
        }
    mIHkernel->updateWrittenImage();

    // Normalize the kernel to its energy
    for (uint32_t ix = 0; ix < kernelSize; ix++)
        for (uint32_t iy = 0; iy < kernelSize; iy++)
            mIHkernel->write(mIHkernel->read(ix, iy)/kernelSum, ix, iy);
    mIHkernel->updateWrittenImage();
}

}