#include "SGR_Recorder_interface.h"
#include "SGR_Recorder.hpp"

extern "C"
{
    SGRRHandle create_SGR_Recorder(float pxSize, float mlaPitch, float mlaDist)
    {
        return new SGR_Recorder(pxSize, mlaPitch, mlaDist);
    }

    void free_SGR_Recorder(SGRRHandle p)
    {
        delete (SGR_Recorder*) p;
    }

    const char* get_SGR_state_descr(SGRRHandle p)
    {
        return ((SGR_Recorder*) p)->getStateDescription();
    }
}
