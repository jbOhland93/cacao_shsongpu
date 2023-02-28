#include "SGR_Recorder.hpp"
#include "milkDebugTools.h"

extern "C"
{

SGR_Recorder::SGR_Recorder(IMAGE* in, IMAGE* dark, float pxSize, float mlaPitch, float mlaDist)
    :mpInput(in), mpDark(dark), mPxSize(pxSize), mMlaPitch(mlaPitch), mMlaDist(mlaDist)
{
    mApertureDiameter = mlaPitch/mPxSize;

    // Check if input and dark images are compatible
    if (mpInput->md->naxis != 2 || mpDark->md->naxis != 2)
    {
        mState = RECSTATE::ERROR;
        mErrDescr = "Input- and dark image have to be two-dimensional.";
    }
    else
    {
        uint32_t* insize = mpInput->md->size;
        uint32_t* darksize = mpDark->md->size;
        if (insize[0] != darksize[0] || insize[1] != darksize[1])
        {
            mState = RECSTATE::ERROR;
            mErrDescr = "Input- and dark image are not of the same size.";
        }
    }
}

errno_t SGR_Recorder::sampleDo()
{
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

}