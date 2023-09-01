#include "SpotFitter.hpp"
#include <algorithm>
#include <sstream>

SpotFitter::SpotFitter(spImageHandler(float) imageHandler)
    : mImageHandler(imageHandler)
{
}

void SpotFitter::setLineoutWidth(uint32_t width)
{
    if (width < 1)
    {
        printf("SpotFitter::setLineoutWidth: Width must be > 0.\n");
        return;
    }
    else if (width >= mLineoutLength)
    {
        printf("SpotFitter::setLineoutWidth: Width must be < ROI size.\n");
        return;
    }
    else
        mLineoutWidth = width;
}

void SpotFitter::makeLineouts(bool forceFullWidth,
                std::string streamPrefix,
                bool visualize)
{
    if (mSpotROIs.size() == 0)
    {
        printf("SpotFitter::makeLineouts: No ROIs set. Aborting...\n");
        return;
    }
    
    // Determin the width for the lineout
    uint32_t lineoutWidth = mLineoutWidth;
    if (forceFullWidth || lineoutWidth < 1)
        lineoutWidth = mLineoutLength;
    
    // Repopulate the lineout vectors
    mLineoutsX.clear();
    mLineoutsY.clear();

    for (int i = 0; i < mSpotROIs.size(); i++)
    {
        mImageHandler->setROI(mSpotROIs.at(i));
        // Initialize the lineout arrays
        std::shared_ptr<double[]> lineoutX(new double[mLineoutLength]);
        std::shared_ptr<double[]> lineoutY(new double[mLineoutLength]);
        for (int i = 0; i < mLineoutLength; i++)
        {
            lineoutX[i] = 0;
            lineoutY[i] = 0;
        }
        // Calculate the lineouts
        uint32_t lineoutOffset = (mLineoutLength - lineoutWidth)/2;
        for (int ix = 0; ix < mLineoutLength; ix++)
            for (int iy = 0; iy < lineoutWidth; iy++)
                lineoutX[ix] += mImageHandler->read(ix, iy+lineoutOffset);
        mLineoutsX.push_back(lineoutX);
        for (int ix = 0; ix < lineoutWidth; ix++)
            for (int iy = 0; iy < mLineoutLength; iy++)
                lineoutY[iy] += mImageHandler->read(ix+lineoutOffset, iy);
        mLineoutsY.push_back(lineoutY);
    }

    // Create an image of the lineouts, if desired
    if (visualize)
    {
        std::string name = streamPrefix;
        name.append("spotLO");
        spImageHandler(double) lineoutIH =
            ImageHandler<double>::newImageHandler(
                name,
                mLineoutLength * 2,
                mSpotROIs.size());
        lineoutIH->setPersistent(true);
        
        for (int i = 0; i < mSpotROIs.size(); i++)
            for (int j = 0; j < mLineoutLength; j++)
            {
                lineoutIH->write(mLineoutsX.at(i)[j], j, i);
                lineoutIH->write(mLineoutsY.at(i)[j], j+mLineoutLength, i);
            }
        mImageHandler->updateWrittenImage();
        printImStreamDescr(name, "lineouts");
    }
}

void SpotFitter::fitCurves(std::string streamPrefix, bool visualize)
{
    if (mLineoutsX.size() == 0)
    {
        printf("SpotFitter::fitCurves: No lineouts found. Aborting...\n");
        return;
    }

    // Prepare image handlers for fit visualization if desired
    spImageHandler(double) fitIH = nullptr;
    spImageHandler(double) errIH = nullptr;
    if (visualize)
    {

        std::string name = streamPrefix;
        name.append("spotLnOuzFits");
        fitIH = ImageHandler<double>::newImageHandler(
                name,
                mLineoutLength * 2,
                mSpotROIs.size());
        fitIH->setPersistent(true);
        printImStreamDescr(name, "spot fits");

        name = streamPrefix;
        name.append("LnOutFitErrs");
        errIH = ImageHandler<double>::newImageHandler(
                name,
                mLineoutLength * 2,
                mSpotROIs.size());
        errIH->setPersistent(true);
        printImStreamDescr(name, "fit errors");
    }

    // Create initial parameter sets
    OneDGaussianFit::gaussianParams pXinit;
    pXinit.amplitude = *std::max_element(
        mLineoutsX.at(0).get(),
        mLineoutsX.at(0).get()+mLineoutLength);
    pXinit.mean = mLineoutLength/2.;
    pXinit.offset = 0;
    pXinit.stddev = mLineoutLength/8.;
    OneDGaussianFit::gaussianParams pYinit;
    pYinit.amplitude = *std::max_element(
        mLineoutsY.at(0).get(),
        mLineoutsY.at(0).get()+mLineoutLength);
    pYinit.mean = mLineoutLength/2.;
    pYinit.offset = 0;
    pYinit.stddev = mLineoutLength/8.;

    // Repopulate the fit result vectors
    mFitResultsX.clear();
    mFitResultsY.clear();

    // Do fit for all lineouts
    for (size_t i = 0; i < mLineoutsX.size(); i++)
    {
        // Prepare fit- and visualization arrays
        OneDGaussianFit::gaussianParams pX;
        OneDGaussianFit::gaussianParams pY;
        std::shared_ptr<double[]> recrX = nullptr;
        std::shared_ptr<double[]> recrY = nullptr;
        if (visualize)
        {
            std::shared_ptr<double[]> tmpX(new double[mLineoutLength]);
            recrX = tmpX;
            std::shared_ptr<double[]> tmpY(new double[mLineoutLength]);
            recrY = tmpY;
        }
        
        // Do fits
        OneDGaussianFit::fitGaussian(
            mLineoutsX.at(i),
            mLineoutLength,
            pXinit,
            &pX,
            recrX);
        mFitResultsX.push_back(pX);
        OneDGaussianFit::fitGaussian(
            mLineoutsY.at(i),
            mLineoutLength,
            pYinit,
            &pY,
            recrY);
        mFitResultsY.push_back(pY);

        // Store fir recreation arrays for visualization
        if (visualize)
            for (int j = 0; j < mLineoutLength; j++)
            {
                fitIH->write(recrX[j], j, i);
                fitIH->write(recrY[j], j+mLineoutLength, i);
                errIH->write(mLineoutsX.at(i)[j]-recrX[j], j, i);
                errIH->write(mLineoutsY.at(i)[j]-recrY[j], j+mLineoutLength, i);
            }
    }
    
    if (visualize)
    {   // Update the visualization images
        fitIH->updateWrittenImage();
        errIH->updateWrittenImage();
    }
}

double SpotFitter::getAvgStdDev()
{
    if (mFitResultsX.size() == 0)
    {
        printf("SpotFitter::getAvgStdDev: No fit results found. Aborting...\n");
        return -1;
    }

    double sum = 0;
    double totalWeight = 0;
    for (size_t i = 0; i < mFitResultsX.size(); i++)
    {
        // The stdDev can be negative (sign is undetermined due to sqr)
        // Therefore: Take absolute value.
        double stddevX = abs(mFitResultsX.at(i).stddev);
        double stddevY = abs(mFitResultsY.at(i).stddev);
        double weightX = mFitResultsX.at(i).amplitude;
        double weightY = mFitResultsY.at(i).amplitude;
        sum += stddevX * weightX + stddevY * weightY;
        totalWeight += weightX + weightY;
    }
    return sum/totalWeight;
}

double SpotFitter::getAvgOffset()
{
    if (mFitResultsX.size() == 0)
    {
        printf("SpotFitter::getAvgOffset: No fit results found. Aborting...\n");
        return -1;
    }

    double sum = 0;
    double totalWeight = 0;
    for (size_t i = 0; i < mFitResultsX.size(); i++)
    {
        double weightX =  mFitResultsX.at(i).amplitude;
        sum += mFitResultsX.at(i).offset * weightX;
        double weightY =  mFitResultsY.at(i).amplitude;
        sum += mFitResultsY.at(i).offset * weightY;
        totalWeight += weightX + weightY;
    }
    return sum/totalWeight;
}

std::vector<Point<double>> SpotFitter::getFittedSpotCenters()
{
    if (mFitResultsX.size() == 0)
    {
        printf("SpotFitter::getFittedSpotCenters: No fit results found. Aborting...\n");
        return std::vector<Point<double>>();
    }

    std::vector<Point<double>> centers;
    for (size_t i = 0; i < mFitResultsX.size(); i++)
    {
        Point<uint32_t> root = mSpotROIs.at(i).mRoot;
        double offX = mFitResultsX.at(i).mean;
        double ampX = mFitResultsX.at(i).amplitude;
        double offY = mFitResultsY.at(i).mean;
        double ampY = mFitResultsY.at(i).amplitude;
        Point<double> offset(offX, offY);
        Point<double> center = offset + root;
        center.mIntensity = (ampX+ampY)/2.;
        centers.push_back(center);
    }
    return centers;
}

void SpotFitter::expressFit(
        std::vector<Point<uint32_t>> centers,
        uint32_t windowSize,
        uint32_t iterations,
        std::string streamPrefix,
        bool visualize)
{
    if (iterations < 1)
        return;
    
    bool visualizeFirst = (iterations == 1) && visualize;

    setROIs(centers, windowSize);
    makeLineouts(false, streamPrefix, visualizeFirst);
    fitCurves(streamPrefix, visualizeFirst);

    for (int i = 1; i < iterations; i++)
    {
        // If visualization is enabled, do it only in last iteration
        bool visualizeCurrent = (i+1 == iterations) && visualize;

        // Get last fit results
        double stdDev = getAvgStdDev();
        std::vector<Point<double>> fittedCenters = getFittedSpotCenters();

        setROIs(fittedCenters, windowSize);
        // Only include 2 stdDeviations in each direction in lineout
        setLineoutWidth(ceil(4*stdDev));
        makeLineouts(false, streamPrefix, visualizeCurrent);
        fitCurves(streamPrefix, visualizeCurrent);
    }
}

void SpotFitter::printImStreamDescr(
    std::string imName,
    std::string contentShort)
{
    std::stringstream ss;
    ss << "Spot fitter: see image stream '";
    ss << imName;
    ss << "' for the " << contentShort << ".\n";
    ss << "\tY = spot index\n";
    ss << "\tX = lineout direction - ";
    ss << "left column is X, right column is Y.\n";
    printf("%s", ss.str().c_str());
}
