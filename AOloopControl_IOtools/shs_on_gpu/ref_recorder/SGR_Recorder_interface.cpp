#include "SGR_Recorder_interface.h"
#include "SGR_Recorder.hpp"

extern "C"
{
    SGRRHandle create_SGR_Recorder(
        IMAGE* in,
        IMAGE* dark,
        float pxSize,
        float mlaPitch,
        float mlaDist,
        const char* streamPrefix,
        int64_t visualize)
    {
        return new SGR_Recorder(
            in,
            dark,
            pxSize,
            mlaPitch,
            mlaDist,
            streamPrefix,
            visualize > 0);
    }

    void free_SGR_Recorder(SGRRHandle p)
    {
        delete (SGR_Recorder*) p;
    }

    errno_t SGRR_sample_do(SGRRHandle p)
    {
        return ((SGR_Recorder*) p)->sampleDo();
    }

    const char* get_SGRR_state_descr(SGRRHandle p)
    {
        return ((SGR_Recorder*) p)->getStateDescription();
    }
}
