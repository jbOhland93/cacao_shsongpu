#ifndef SGE_RESHAPER_INTERFACE_H
#define SGE_RESHAPER_INTERFACE_H

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the reshaper object to be used for function calls
typedef void * SGEReshapeHandle;
// Constructor
SGEReshapeHandle create_SGE_Reshaper(
    IMAGE* input,           // Stream with SHS reference positions
    IMAGE* mask,            // Stream of the SHS pupil mask
    int64_t linesAsSlices); // Reshape lines into slices instead of stacked if > 0
    
// Desctructor
void free_SGE_Reshaper(SGEReshapeHandle);

// Do reshaping on available data
errno_t SGEE_reshape_do(SGEReshapeHandle);

#ifdef __cplusplus
}
#endif

#endif // SGE_RESHAPER_INTERFACE_H