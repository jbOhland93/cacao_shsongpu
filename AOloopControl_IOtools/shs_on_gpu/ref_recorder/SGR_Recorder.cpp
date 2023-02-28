#include "SGR_Recorder.hpp"
#include <cstdio>

extern "C"
{

SGR_Recorder::SGR_Recorder(float pxSize, float mlaPitch, float mlaDist)
    :mPxSize(pxSize), mMlaPitch(mlaPitch), mMlaDist(mlaDist)
{
}

const char* SGR_Recorder::getStateDescription()
{
    switch(mState) {
    case RECSTATE::INIT: return "Initializing...\n";
    case RECSTATE::RECORD: return "Recording...\n";
    case RECSTATE::FINISH: return "Done!\n";
    default: return "UNKNOWN\n";
    }
}

}