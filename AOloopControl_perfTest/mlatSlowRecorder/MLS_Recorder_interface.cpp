#include "MLS_Recorder_interface.h"
#include "MLS_Recorder.hpp"

extern "C"
{

    MLSRHandle create_MLS_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps, // process relatef fps
        IMAGE* dmstream,            // Stream of the DM input
        IMAGE* wfsstream,           // Stream of the WFS output
        int64_t skipMFramerate,     // If true, the FPS measurement prior to the latency is skipped
        float fpsMeasTime,          // Timeframe for wfs framerate estimation
        int32_t pokePattern,        // Poke pattern
        const char* shmPatternName, // Name of pattern stream in shm
        uint32_t shmImPatternIdx,   // Index of the shm pattern slice to be poked
        float maxActStroke,         // Maximum actuator stroke in pattern
        uint32_t numPokes,          // number of iterations
        uint32_t framesPerPoke,     // number of frames per iteration
        int64_t saveRaw)            // If true, each iterations frames is saved to fits
    {
        return new MLS_Recorder(
            fps,
            dmstream,
            wfsstream,
            skipMFramerate > 0,
            fpsMeasTime,
            pokePattern,
            shmPatternName,
            shmImPatternIdx,
            maxActStroke,
            numPokes,
            framesPerPoke,
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

}
