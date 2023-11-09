#ifndef PUPILZERNIKEGENERATOR_INTERFACE_H
#define PUPILZERNIKEGENERATOR_INTERFACE_H

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to the zernike generator object to be used for function calls
typedef void * PupilZernGenHandle;
// Constructor
PupilZernGenHandle create_PupilZernGenerator(
    IMAGE* zerncoeff,       // Stream with zernike coefficients
    IMAGE* mask);           // Stream of the SHS pupil mask
    
// Desctructor
void free_PupilZernGen(PupilZernGenHandle);

// Do expansion on available data
errno_t PupilZernGen_expand_do(
    PupilZernGenHandle, // Pointer to generator object
    int64_t output2D);  // Expand to 2D pupil instead of 1D if > 0

#ifdef __cplusplus
}
#endif

#endif // PUPILZERNIKEGENERATOR_INTERFACE_H