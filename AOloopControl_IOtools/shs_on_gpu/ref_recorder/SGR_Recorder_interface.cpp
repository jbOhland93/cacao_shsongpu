#include "SGR_Recorder_interface.h"
#include "SGR_Recorder.hpp"

extern "C"
{
    SGRRHandle create_SGR_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps,
        IMAGE* in,
        IMAGE* dark,
        float pxSize,
        float mlaPitch,
        float mlaDist,
        uint32_t numSamples,
        int64_t visualize)
    {
        return new SGR_Recorder(
            fps,
            in,
            dark,
            pxSize,
            mlaPitch,
            mlaDist,
            numSamples,
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

    errno_t SGRR_evaluate_rec_buffers(
        SGRRHandle p,
        float minRelIntensity,
        float uradPrecisionThresh)
    {
        return ((SGR_Recorder*) p)->evaluateRecBuffers(
            minRelIntensity, uradPrecisionThresh);
    }

    const char* get_SGRR_state_descr(SGRRHandle p)
    {
        return ((SGR_Recorder*) p)->getStateDescription();
    }
}
