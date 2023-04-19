#ifndef CUDA_SGE_UTIL
#define CUDA_SGE_UTIL

#define CUDA_CALLABLE_MEMBER __host__ __device__

#define printCE(err) \
if (err != cudaSuccess) printf("CU ERR: %s\n", cudaGetErrorString(err));

#endif // CUDA_SGE_UTIL