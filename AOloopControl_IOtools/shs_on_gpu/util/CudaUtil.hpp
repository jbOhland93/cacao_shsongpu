#ifndef CUDA_SGE_UTIL
#define CUDA_SGE_UTIL

#include <cublas_v2.h>

#define CUDA_CALLABLE_MEMBER __host__ __device__

#define printCE(err) \
if (err != cudaSuccess) printf("CU ERR: %s\n", cudaGetErrorString(err));

#define printCuBE(err) \
if (err != CUBLAS_STATUS_SUCCESS) printf("cuBLAS ERR: %s\n", cublasGetErrorString(err));

inline const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

#endif // CUDA_SGE_UTIL