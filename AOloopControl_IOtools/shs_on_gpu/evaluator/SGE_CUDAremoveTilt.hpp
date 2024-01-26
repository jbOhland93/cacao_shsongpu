#include <cuda.h>
#include "SGE_GridLayout.hpp"

// This code has been adapted from Justin Luitjens' post:
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

// Reduce
__inline__ __device__
int warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__
int blockReduceSum(float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane==0)
        shared[wid]=val;    // Write reduced value to shared memory

    __syncthreads();        // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0)
        val = warpReduceSum(val); //Final reduce within first warp

    return val;
}


__global__ void deviceReduceKernel(float *in, float* out, int N) {
  float sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

__global__ void deviceRemoveTilt(float *grad, float* reducedGrad, int numSamples)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Calc average
    float avg;
    if (index < numSamples)
        avg = reducedGrad[0]/numSamples;
    else
        avg = reducedGrad[numSamples]/numSamples;
    // Subtract average
    if (index < numSamples*2)
        grad[index] -= avg;
}
