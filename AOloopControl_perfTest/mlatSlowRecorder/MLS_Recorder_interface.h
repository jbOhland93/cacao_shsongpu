#ifndef MLS_RECORDER_INTERFACE_H
#define MLS_RECORDER_INTERFACE_H

#include "../../../../src/CommandLineInterface/CLIcore.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the recorder object to be used for function calls
typedef void * MLSRHandle;
// Constructor
MLSRHandle create_MLS_Recorder(
    IMAGE* dmstream,        // Stream of the DM input
    IMAGE* wfsstream,       // Stream of the WFS output
    float fpsMeasTime,      // Timeframe for wfs framerate estimation
    uint32_t pokePattern,   // Poke pattern
    float maxActStroke,     // Maximum actuator stroke in pattern
    uint32_t numPokes,      // number of iterations
    uint32_t framesPerPoke, // number of frames per iteration
    FUNCTION_PARAMETER_STRUCT* fps, // process relatef fps
    int64_t saveRaw);       // If true, each iterations frames is saved to fits

// Desctructor
void free_MLS_Recorder(MLSRHandle p);

// Launch latency recording routine
void mlsRecordDo(MLSRHandle p);

// Result getters
float getFPS_Hz(MLSRHandle p);
float getHWdelay_frames(MLSRHandle p);
float getHWdelay_us(MLSRHandle p);
float getRiseTime0to90_frames(MLSRHandle p);
float getRiseTime0to90_us(MLSRHandle p);
float getHWlatency_frames(MLSRHandle p);
float getHWlatency_us(MLSRHandle p);

#ifdef __cplusplus
}
#endif

#endif // MLS_RECORDER_INTERFACE_H