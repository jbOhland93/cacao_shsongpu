#include "MLS_Recorder_interface.h"
#include "MLS_Recorder.hpp"

extern "C"
{

    MLSRHandle create_MLS_Recorder(
        IMAGE* dmstream,        // Stream of the DM input
        IMAGE* wfsstream,       // Stream of the WFS output
        float fpsMeasTime,      // Timeframe for wfs framerate estimation
        uint32_t pokePattern,   // Poke pattern:
        float maxActStroke,     // Maximum actuator stroke in pattern
        uint32_t numPokes,      // number of iterations
        uint32_t framesPerPoke, // number of frames per iteration
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        int64_t saveRaw)        // If true, each iterations frames is saved to fits
    {
        return new MLS_Recorder(
            dmstream,
            wfsstream,
            fpsMeasTime,
            pokePattern,
            maxActStroke,
            numPokes,
            framesPerPoke,
            fps,
            saveRaw > 0);
    }

    void free_MLS_Recorder(MLSRHandle p)
    {
        delete (MLS_Recorder*) p;
    }

    void mlsRecordDo(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->recordDo();
    }

    // Getters

    float getFPS_Hz(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getFPS_Hz();
    }

    float getHWdelay_frames(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getHWdelay_frames();
    }
    
    float getHWdelay_us(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getHWdelay_us();
    }

    float getRiseTime0to90_frames(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getRiseTime0to90_frames();
    }
    
    float getRiseTime0to90_us(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getRiseTime0to90_us();
    }

    float getHWlatency_frames(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getHWlatency_frames();
    }

    float getHWlatency_us(MLSRHandle p)
    {
        return ((MLS_Recorder*) p)->getHWlatency_us();
    }

}
