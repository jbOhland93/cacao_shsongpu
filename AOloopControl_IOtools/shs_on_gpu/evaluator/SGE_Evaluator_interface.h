#ifndef SGE_EVALUATOR_INTERFACE_H
#define SGE_EVALUATOR_INTERFACE_H

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the evaluator object to be used for function calls
typedef void * SGEEHandle;
// Constructor
SGEEHandle create_SGE_Evaluator(
    IMAGE* ref,                 // Stream with SHS reference positions
    IMAGE* shscam,              // Stream of the SHS camera
    IMAGE* shsdark,             // Stream containing the darkframe for the SHS camera
    const char* streamPrefix);  // Prefix for the ISIO streams
    
// Desctructor
void free_SGE_Evaluator(SGEEHandle);

// Do evaluation on available data
errno_t SGEE_eval_do(
    SGEEHandle p,
    int64_t useAbsRef,     // Reference will be absolute w.r.t. the MLA grid
    int64_t calcWF,        // Calculate the WF from the gradient field
    int64_t cpyGradToCPU,  // Copy the evaluated gradient to the CPU
    int64_t cpyWfToCPU,    // Copy the WF to the CPU, if reconstructed
    int64_t cpyIntToCPU);  // Copy the intensity to the CPU

#ifdef __cplusplus
}
#endif

#endif // SGR_RECORDER_INTERFACE_H