struct SGR_Recorder;

#ifdef __cplusplus
extern "C" {
#endif

typedef void * SGRRHandle;
SGRRHandle create_SGR_Recorder(float pxSize, float mlaPitch, float mlaDist);
void free_SGR_Recorder(SGRRHandle);

const char* get_SGR_state_descr(SGRRHandle);

#ifdef __cplusplus
}
#endif
