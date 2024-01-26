#ifndef SGR_RECORDER_HPP
#define SGR_RECORDER_HPP

#include "SGR_ReferenceKW.hpp"
#include "../util/ImageHandler2D.hpp"
#include "../util/GaussianKernel.hpp"
#include <errno.h>
#include <string>
#include <memory>
#include <vector>


// A class for recording SHS references
class SGR_Recorder
{
public:
    // Ctor, doing the initialization
    SGR_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        IMAGE* in,           // Raw camera stream
        IMAGE* dark,         // Stream holding a dark for subtraction
        float pxSize_um,     // Size of the camera pixels in um
        float mlaPitch_um,   // Distance of the microlenses in um
        float mlaDist_um,    // Distance of the microlenses to the cam chip in um
        uint32_t numSamples, // number of samples to be recorded
        bool visualize = false); // If true, additional streams for
                                 // visual testing are generated

    // Triggers reading the input- and dark stream
    errno_t sampleDo();
    // Evaluates the recorded buffers and generates the reference output
    // minRelIntensity: Threshhold for generating the spot mask
    //      If the intensity of the subaperture is lower
    //      than the max. intensity multiplied by this value,
    //      the subaperture is regarded invalid.
    // uradPrecisionThresh: Threshhold for generating the spot mask
    //      If the precision of a subaperture is better than this,
    //      the sample will be included in the mask.
    errno_t evaluateRecBuffers(
        float minRelIntensity,
        float uradPrecisionThresh);
    // Returns a description of the current internal state
    const char* getStateDescription();

private:
    FUNCTION_PARAMETER_STRUCT* mp_fps;
    // GPU ID
    int mDevice = -1;
    // Internal status
    // Call getStateDescription to get printable info
    enum RECSTATE {
        ERROR,
        INITIALIZING,
        AWAIT_SAMPLE,
        SAMPLING,
        READY_FOR_EVAL,
        EVALUATING,
        FINISH};
    RECSTATE mState = RECSTATE::INITIALIZING;
    uint32_t mSamplesExpected;  // Number of samples promised by ctor call
    uint32_t mSamplesAdded = 0; // For calculating the average at the end

    // If visualization is true, additional output images will
    // be writen to SHM in order to verify the process
    bool mVisualize = true;
    std::string mTeststreamPrefix = "Vrfy-"; // Prefix for visualization streams

// SHS parameters
    float mPxSize_um;
    float mMlaPitch_um;
    float mMlaDist_um;
    float mApertureDiameter_px;
    double mShiftToGradConstant;

// Intermediary results
    spGKernel mpKernel;
    
// Image streams / ImageHandlers
    IMAGE* mpInput;
    IMAGE* mpDark;
    // Properties of the input image stream, for easier access
    uint32_t mImgWidth;
    uint32_t mImgHeight;
    uint32_t mNumPixels;

    // Stream for the result of mpInput-mpDark
    spImHandler2D(float) mIHdarkSubtract;
    // Binary image, thresholded from darkSubtract
    spImHandler2D(uint8_t) mIHthresh;
    // Stream for erosion of thresh to estimate spot positions
    spImHandler2D(uint8_t) mIHerode;
    // Stream for the convolution of darkSubtract and kernel
    spImHandler2D(float) mIHconvolution;
    // Recording stream for the spot intensities
    spImHandler2D(float) mIHintensityREC;
    // Recording stream for the spot positions
    spImHandler2D(float) mIHposREC;
    // A display for the search grid. Only used if mVisualize == true.
    spImHandler2D(float) mIHgridVisualization;

// Preliminary search grid, used for calibration
    Point<uint32_t> mGridSize;
    uint32_t mGridRectSize;
    std::vector<std::vector<Rectangle<uint32_t>>> mGrid;

// String members for persistence of messages, names etc.
    std::string mErrDescr{ "" };
    std::string mStateDescr{ "" };
    std::string mTmpStreamName{ "" };

// Helper functions ==
    // Prepends the test stream prefix to the given name
    const char* makeTestStreamname(const char* name);
    // Analyzes the first frame and sets up the analysis
    void prepareSpotFinding();
    // Removes points from pIn which are too close
    std::vector<Point<uint32_t>> filterByMinDistance(
        std::vector<Point<uint32_t>> pIn,
        double minDistance);
    // Spans a regular grid, matching the given unsorted spots
    void spanCoarseGrid(std::vector<Point<double>> fitSpots);
};

#endif // SGR_RECORDER_HPP
