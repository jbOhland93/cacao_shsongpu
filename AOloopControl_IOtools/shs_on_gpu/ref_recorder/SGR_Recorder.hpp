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
    SGR_Recorder(
        IMAGE* in,
        IMAGE* dark,
        float pxSize,
        float mlaPitch,
        float mlaDist,
        const char* streamPrefix = "");
    errno_t sampleDo();
    const char* getStateDescription();

private:
    // If visualization is true, output images will
    // be writen to SHM in order to verify the process
    bool mVisualize = true;
    float mPxSize;
    float mMlaPitch;
    float mMlaDist;
    float mApertureDiameter;
    std::string mStreamPrefix;
    uint32_t mImgWidth;
    uint32_t mImgHeight;
    uint32_t mNumPixels;

    IMAGE* mpInput;
    IMAGE* mpDark;
    spImageHandler(float) mIHdarkSubtract;
    spImageHandler(uint8_t) mIHthresh;
    spImageHandler(uint8_t) mIHerode;
    Point<uint32_t> mGridSize;
    uint32_t mGridRectSize;
    std::vector<std::vector<Rectangle<uint32_t>>> mGrid;
    spImageHandler(float) mIHgridVisualization;

    enum RECSTATE {ERROR,INIT,RECORD,FINISH};
    RECSTATE mState = RECSTATE::INIT;
    std::string mErrDescr{ "" };

    std::string makeErrorMessage();
    const char* makeStreamname(const char* name);
    std::vector<Point<uint32_t>> filterByMinDistance(
        std::vector<Point<uint32_t>> pIn,
        double minDistance);
    void spanCoarseGrid(std::vector<Point<double>> fitSpots);
};

#endif // SGR_RECORDER_HPP
