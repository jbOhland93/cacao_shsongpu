#include "SGR_Recorder.hpp"
#include "../util/SpotFitter.hpp"

extern "C" {
    #include "../../../../../src/CommandLineInterface/CLIcore.h"
    #include "../../../../../src/COREMOD_iofits/savefits.h"
    #include "../../../../../src/COREMOD_memory/read_shmim.h"
}

#include <string>
#include <iostream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C"
{

SGR_Recorder::SGR_Recorder(
    IMAGE* in,
    IMAGE* dark,
    float pxSize_um,
    float mlaPitch_um,
    float mlaDist_um,
    uint32_t numSamples,
    const char* savingLocation,
    bool visualize)
    :
    mpInput(in),
    mpDark(dark),
    mPxSize_um(pxSize_um),
    mMlaPitch_um(mlaPitch_um),
    mMlaDist_um(mlaDist_um),
    mSamplesExpected(numSamples),
    mSavingLocation(savingLocation),
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
            // Initialize parameters
            mApertureDiameter_px = mlaPitch_um/mPxSize_um;
            mGridRectSize = floor(mApertureDiameter_px);
            // Calculate the SHS device constant
            // This constant can transfer the shift of each SHS-spot in pixels
            // to a local gradient via multiplication, i.e. the WF variation
            // in micrometers over the current subaperture.
            mShiftToGradConstant = mPxSize_um * mMlaPitch_um / mMlaDist_um;

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
            mIHdarkSubtract->getImage(),
            mpKernel->getKernelIH()->getImage());

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
        // Prepare final reference names
        std::string refBaseName = mpInput->name;
        refBaseName.append("_");
        std::string refNameSuffix = "RefPositions";
        std::string maskNameSuffix = "RefMask";
        std::string intensityNameSuffix = "RefIntensity";
        std::string refName = refBaseName;
        std::string maskName = refBaseName;
        std::string intensityName = refBaseName;
        refName.append(refNameSuffix);
        maskName.append(maskNameSuffix);
        intensityName.append(intensityNameSuffix);

        // Initialize the  image handlers for the evaluation
        spImHandler2D(float) IHavgI = ImageHandler2D<float>::newImageHandler2D(
            intensityName, mGridSize.mX, mGridSize.mY, 1, 10);
        IHavgI->setPersistent(true);
        spImHandler2D(float) IHavgP = ImageHandler2D<float>::newImageHandler2D(
            "7_eval-AVGpos", mGridSize.mX*2, mGridSize.mY);
        IHavgP->setPersistent(mVisualize);
        spImHandler2D(float) IHstdDvP = ImageHandler2D<float>::newImageHandler2D(
            "7_eval-STDDVpos", mGridSize.mX*2, mGridSize.mY);
        IHstdDvP->setPersistent(mVisualize);
        spImHandler2D(uint8_t) IHspotMask = ImageHandler2D<uint8_t>::newImageHandler2D(
            maskName, mGridSize.mX, mGridSize.mY, 1, 10);
        IHspotMask->setPersistent(true);

        // Calculate the stability threshold for the mask
        float deflectionPrecision = mMlaDist_um * tan(uradPrecisionThresh*1e-6);
        float pxPrecision = deflectionPrecision / mPxSize_um;
        printf("Demanded spot stability: %.1f µrad == %.4f px.\n",
            uradPrecisionThresh, pxPrecision);
        
        // Size of the circular buffer == # of frames recorded
        uint32_t cbSize = mIHintensityREC->getImage()->md->CBsize;
        // Initialize fields for statistics
        std::vector<float> vI, vX, vY;
        float avgI, avgX, avgY, stdDvX, stdDvY, stdDvXY;
        
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
                    IHspotMask->write(1, ix, iy);
                else
                    IHspotMask->write(0, ix, iy);
            }

        // Erode mask edge
        // 3 is only removing
        //  - outliers
        //  - pixels sticking out from corners
        // 4 will also erode 
        //  - corners
        //  - pixels sticking out alone from straight edges
        int numOfValidSpots = IHspotMask->erode(4, false);

        // Update images
        IHavgI->updateWrittenImage();
        IHavgP->updateWrittenImage();
        IHstdDvP->updateWrittenImage();
        IHspotMask->updateWrittenImage();
        printf("Intensity map generated.\n\t=> Stream name: %s\n", IHavgI->getImage()->name);
        printf("Mask generated. %d valid subapertures detected.\n", numOfValidSpots);
        printf("\t=> Stream name: %s\n", IHspotMask->getImage()->name);

        // == Make reference ==

        // The reference is an image with 2 lines:
        // First line holds the average X positions of the spots within the mask
        // Second line holds the average Y positions of the spots within the mask
        spImHandler2D(float) IHcpuRef = ImageHandler2D<float>::newImageHandler2D(
            refName, numOfValidSpots, 2, 1, 10);
        IHcpuRef->setPersistent(true);

        // Store the (valid) average positions in the reference stream
        int numWritten = 0;
        for (uint32_t ix = 0; ix < mGridSize.mX; ix++)
            for (uint32_t iy = 0; iy < mGridSize.mY; iy++)
                if (IHspotMask->read(ix, iy))
                {
                    float posX = IHavgP->read(ix, iy);
                    float posY = IHavgP->read(ix+mGridSize.mX, iy);
                    IHcpuRef->write(posX, numWritten, 0);
                    IHcpuRef->write(posY, numWritten, 1);
                    numWritten++;
                }
        IHcpuRef->updateWrittenImage();
        printf("Reference generated.\n\t=> Stream name: %s\n", IHcpuRef->getImage()->name);

        // Set keywords to streams - include relevant metadata
        std::vector<std::shared_ptr<ImageHandler2DBase>> IHs;
        IHs.push_back(std::static_pointer_cast<ImageHandler2DBase>(IHavgI));
        IHs.push_back(std::static_pointer_cast<ImageHandler2DBase>(IHspotMask));
        IHs.push_back(std::static_pointer_cast<ImageHandler2DBase>(IHcpuRef));
        for (int i = 0; i < IHs.size(); i++)
        {
            try
            {
                IHs.at(i)->setKeyword(0, REF_KW_KERNEL_STDDEV, (double) mpKernel->getStdDev());
                IHs.at(i)->setKeyword(1, REF_KW_KERNEL_SIZE, (int64_t) mpKernel->getKernelSize());
                std::string inName = std::string(mpInput->name);
                if (inName.length() > 16)
                    inName = inName.substr(0, 16);
                IHs.at(i)->setKeyword(2, REF_KW_INPUT_NAME, inName);
                std::string darkName = std::string(mpDark->name);
                if (darkName.length() > 16)
                    darkName = darkName.substr(0, 16);
                IHs.at(i)->setKeyword(3, REF_KW_DARK_NAME, darkName);
                int64_t suffixLen = (int64_t) (std::string(IHs.at(i)->getImage()->name).length() - refBaseName.length());
                IHs.at(i)->setKeyword(4, REF_KW_SUFFIX_LEN, suffixLen);
                IHs.at(i)->setKeyword(5, REF_KW_REF_SUFFIX, refNameSuffix);
                IHs.at(i)->setKeyword(6, REF_KW_MASK_SUFFIX, maskNameSuffix);
                IHs.at(i)->setKeyword(7, REF_KW_INTENSITY_SUFFIX, intensityNameSuffix);
                IHs.at(i)->setKeyword(8, REF_KW_PX_PITCH, (double) mMlaPitch_um / mPxSize_um);
                IHs.at(i)->setKeyword(9, REF_KW_SHIFT_2_GRAD_CONST, mShiftToGradConstant);
            }
            catch(const std::runtime_error& e)
            {
                mErrDescr = "Error during keyword setting: ";
                mErrDescr.append(e.what());
                mState = RECSTATE::ERROR;
                return RETURN_FAILURE;
            }
        }
        printf("Metadata written to ISIO keywords.\n");

        printf("Saving reference fits files ...\n");
        // Guarantee that saving destination exists
        struct stat sb;
        if (!(stat(mSavingLocation.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)))
            if (mkdir(mSavingLocation.c_str(), 0700) != 0)
            {
                mErrDescr = "Error creating saving folder for fits files.";
                mState = RECSTATE::ERROR;
                return RETURN_FAILURE;
            }
        // Save files
        saveImage(IHspotMask);
        saveImage(IHcpuRef);
        saveImage(IHavgI);

        mState = RECSTATE::FINISH;
        printf("\n\n=== SGR_Recorder: Evaluation done! ===\n\n");

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
        mIHdarkSubtract = newImHandler2DFrmIm(float, "1-darksub", mpInput);
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
        mIHthresh = newImHandler2DFrmIm(uint8_t, "2-thresh", mpInput);
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
        mIHerode = newImHandler2DFrmIm(uint8_t, "3-erode", mIHthresh->getImage());
        mIHerode->setPersistent(mVisualize);
        std::vector<Point<uint32_t>> particles;
        while (mIHerode->erode(5, true, &particles) > 0);
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
        printf("Average standard deviation of spot PSF: %.3f\n", spotFitter.getAvgStdDev());
        mpKernel = GaussianKernel::makeKernel(
            spotFitter.getAvgStdDev(), "4-kernel", mVisualize);
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
        mIHconvolution = newImHandler2DFrmIm(float, "5-convol", mpInput);
        mIHconvolution->setPersistent(mVisualize);

        // Prepare the averaging image streams for intensity and positions
        mIHintensityREC = ImageHandler2D<float>::newImageHandler2D(
                "6-recordAmp",
                mGridSize.mX, mGridSize.mY, 1,
                0,
                mSamplesExpected);
        mIHintensityREC->setPersistent(mVisualize);
        mIHposREC = ImageHandler2D<float>::newImageHandler2D(
                "6-recordSpotPos",
                mGridSize.mX*2, mGridSize.mY, 1,
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
        mIHgridVisualization = newImHandler2DFrmIm(float,
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

void SGR_Recorder::saveImage(spIHBase imageHandler)
{
    const char* imageName = imageHandler->getImage()->name;
    // First resolve the image ID
    long id = read_sharedmem_image(imageName);
    if (id >= 0)
    {   // Then, save the file.
        std::string fitsName = generateFitsName(imageName);
        errno_t err = save_fits(imageName, fitsName.c_str());
        if (err == RETURN_SUCCESS)
            printf("\t=> Written to %s\n", fitsName.c_str());
        else
        {
            mErrDescr.append("SGR_Recorder::saveImage: ");
            mErrDescr.append("Error on saving fits file.");
            mState = RECSTATE::ERROR;
        }
    }
    else
    {
        mErrDescr.append("SGR_Recorder::saveImage: ");
        mErrDescr.append("Could not resolve image id before saving fits.");
        mState = RECSTATE::ERROR;
    }
}

std::string SGR_Recorder::generateFitsName(std::string prefix) {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    
    std::tm gmt = *now;
    std::time_t utc_time = mktime(&gmt);

    int offset_minutes = (utc_time - t) / 60;

    // Convert to hours and minutes
    int offset_hours = offset_minutes / 60;
    offset_minutes = std::abs(offset_minutes % 60);

    std::ostringstream oss;
    oss << mSavingLocation << "/" << prefix << "_"
        << (now->tm_year + 1900) << '-'
        << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
        << std::setfill('0') << std::setw(2) << now->tm_mday << "_T"
        << std::setfill('0') << std::setw(2) << now->tm_hour << '.'
        << std::setfill('0') << std::setw(2) << now->tm_min
        << std::setfill('0') << std::setw(3) << std::internal << std::showpos << offset_hours << std::noshowpos << "."
        << std::setfill('0') << std::setw(2) << std::abs(offset_minutes)
        << ".fits";
    
    return oss.str();
}

}