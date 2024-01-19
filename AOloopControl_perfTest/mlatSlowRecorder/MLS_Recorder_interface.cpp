#include "MLS_Recorder_interface.h"
#include "MLS_Recorder.hpp"

extern "C"
{

    MLSRHandle create_MLS_Recorder(
        FUNCTION_PARAMETER_STRUCT* fps,     // process relatef fps
        IMAGE* dmstream,                    // Stream of the DM input
        IMAGE* wfsstream,                   // Stream of the WFS output
        int64_t skipMFramerate,             // If true, the FPS measurement prior to the latency is skipped
        float fpsMeasTime,                  // Timeframe for wfs framerate estimation
        uint32_t numPokes,                  // number of iterations
        uint32_t framesPerPoke,             // number of frames per iteration
        int64_t saveRaw,                    // If true, each iterations frames is saved to fits
        int32_t pokePatternType,            // Poke pattern type
        const char* customPatternStream,    // Name of pattern stream in shm
        uint32_t customPatternSliceIdx,     // Index of the shm pattern slice to be poked
        float patternToStrokeMul,           // Pattern-to-poke factor
        int64_t useCustomResponseStream,    // Don't record the response but use custom one
        const char* customResponseStream,   // Name of response stream in shm
        uint32_t customResponseSliceIdx)    // Index of the shm response slice to be poked
    {
        return new MLS_Recorder(
            fps,
            dmstream,
            wfsstream,
            skipMFramerate > 0,
            fpsMeasTime,
            numPokes,
            framesPerPoke,
            saveRaw > 0,
            pokePatternType,
            customPatternStream,
            customPatternSliceIdx,
            patternToStrokeMul,
            useCustomResponseStream > 0,
            customResponseStream,
            customResponseSliceIdx);
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
