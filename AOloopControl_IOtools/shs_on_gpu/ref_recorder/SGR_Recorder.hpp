#ifndef SGR_RECORDER_HPP
#define SGR_RECORDER_HPP

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>
#include <string>

// A class for recording SHS references
class SGR_Recorder
{
public:
    SGR_Recorder(IMAGE* in, IMAGE* dark, float pxSize, float mlaPitch, float mlaDist);
    errno_t sampleDo();
    const char* getStateDescription();

private:
    float mPxSize;
    float mMlaPitch;
    float mMlaDist;
    float mApertureDiameter;

    IMAGE* mpInput;
    IMAGE* mpDark;

    enum RECSTATE {ERROR,INIT,RECORD,FINISH};
    RECSTATE mState = RECSTATE::INIT;
    std::string mErrDescr{ "" };

    std::string makeErrorMessage();
};

#endif // SGR_RECORDER_HPP