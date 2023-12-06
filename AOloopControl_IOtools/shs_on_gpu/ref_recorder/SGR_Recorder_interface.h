#ifndef SGR_RECORDER_INTERFACE_H
#define SGR_RECORDER_INTERFACE_H

#include "../../../../../src/CommandLineInterface/CLIcore.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the recorder object to be used for function calls
typedef void * SGRRHandle;
// Constructor
SGRRHandle create_SGR_Recorder(
    FUNCTION_PARAMETER_STRUCT* fps, // process relatef fps
    IMAGE* in,                  // Raw camera stream
    IMAGE* dark,                // Stream holding a dark for subtraction
    float pxSize,               // Size of the camera pixels
    float mlaPitch,             // Distance of the microlenses
    float mlaDist,              // Distance of the microlenses to the cam chip
    uint32_t numSamples,        // number of samples to be recorded
    const char* savingLocation, // Prefix for the ISIO streams
    int64_t visualize);         // If true, additional streams for
                                // visual testing are generated
// Desctructor
void free_SGR_Recorder(SGRRHandle);

// Do one sample step
errno_t SGRR_sample_do(SGRRHandle);
// Evaluate the recorded buffers and generates the reference output
// uradPrecisionThresh: Threshhold for generating the spot mask
//      If the precision of a subaperture is better than this,
//      the sample will be included in the mask.
errno_t SGRR_evaluate_rec_buffers(SGRRHandle, float);
// Returns a brief description on the current internal state of the recorder
const char* get_SGRR_state_descr(SGRRHandle);

#ifdef __cplusplus
}
#endif

#endif // SGR_RECORDER_INTERFACE_H