#ifndef SGR_RECORDER_HPP
#define SGR_RECORDER_HPP

#include "SGR_ImageHandler.hpp"
#include <errno.h>
#include <string>
#include <memory>
#include <vector>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// A class for recording SHS references
class SGR_Recorder
{
public:
    // Ctor, doing the initialization
    SGR_Recorder(
        IMAGE* in,      // Raw camera stream
        IMAGE* dark,    // Stream holding a dark for subtraction
        float pxSize,   // Size of the camera pixels
        float mlaPitch, // Distance of the microlenses
        float mlaDist,  // Distance of the microlenses to the cam chip
        const char* streamPrefix = "", // Prefix for the ISIO streams
        bool visualize = false); // If true, additional streams for
                                 // visual testing are generated

    // Triggers reading the input- and dark stream
    errno_t sampleDo();
    // Returns a description of the current internal state
    const char* getStateDescription();

private:
    // Internal status
    enum RECSTATE {ERROR,INIT,READY,SAMPLING,EVALUATING,FINISH};
    RECSTATE mState = RECSTATE::INIT;

    // If visualization is true, additional output images will
    // be writen to SHM in order to verify the process
    bool mVisualize = true;
    std::string mTeststreamPrefix; // Prefix for visualization streams

    // SHS parameters
    float mPxSize;
    float mMlaPitch;
    float mMlaDist;
    float mApertureDiameter;
    
    // Properties of the input image stream, for easier access
    uint32_t mImgWidth;
    uint32_t mImgHeight;
    uint32_t mNumPixels;

    // Image streams / ImageHandlers
    IMAGE* mpInput;
    IMAGE* mpDark;
    std::string mStreamPrefix; // Prefix for all streams to be generated
    spImageHandler(float) mIHdarkSubtract;
    spImageHandler(uint8_t) mIHthresh;
    spImageHandler(uint8_t) mIHerode;
    spImageHandler(float) mIHgridVisualization;
    spImageHandler(float) mIHkernel;
    spImageHandler(float) mIHconvolution;
    spImageHandler(float) mIHIntensityAVG;
    spImageHandler(float) mIHShiftsAVG;
    uint32_t mSamplesAdded = 0; // For calculating the average at the end

    // Preliminary search grid, used for calibration
    Point<uint32_t> mGridSize;
    uint32_t mGridRectSize;
    std::vector<std::vector<Rectangle<uint32_t>>> mGrid;

    // String members for persistence of messages, names etc.
    std::string mErrDescr{ "" };
    std::string mStateDescr{ "" };
    std::string mTmpStreamName{ "" };

    // == Helper functions ==
    // Prepends the stream prefix to the given name
    const char* makeStreamname(const char* name);
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
    // Makes a gaussian kernel with the given standard deviation
    void buildKernel(double stdDev);
};

#endif // SGR_RECORDER_HPP
