#ifndef SGR_RECORDER_INTERFACE_H
#define SGR_RECORDER_INTERFACE_H

#include "ImageStreamIO/ImageStruct.h"
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void * SGRRHandle;
SGRRHandle create_SGR_Recorder(
    IMAGE* in,
    IMAGE* dark,
    float pxSize,
    float mlaPitch,
    float mlaDist,
    const char* streamPrefix,
    int64_t visualize);
void free_SGR_Recorder(SGRRHandle);

errno_t SGRR_sample_do(SGRRHandle);
const char* get_SGRR_state_descr(SGRRHandle);

#ifdef __cplusplus
}
#endif

#endif // SGR_RECORDER_INTERFACE_H