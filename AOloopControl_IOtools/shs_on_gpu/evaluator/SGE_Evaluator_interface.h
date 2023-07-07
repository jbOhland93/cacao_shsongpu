#ifndef SGE_EVALUATOR_INTERFACE_H
#define SGE_EVALUATOR_INTERFACE_H

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the recorder object to be used for function calls
typedef void * SGEEHandle;
// Constructor
SGEEHandle create_SGE_Evaluator(
    IMAGE* ref,         // Stream with SHS reference positions
    IMAGE* shscam,      // Stream of the SHS camera
    IMAGE* shsdark);    // Stream containing the darkframe for the SHS camera
    
// Desctructor
void free_SGE_Evaluator(SGEEHandle);

// Do evaluation on available data
errno_t SGEE_eval_do(SGEEHandle);

#ifdef __cplusplus
}
#endif

#endif // SGR_RECORDER_INTERFACE_H