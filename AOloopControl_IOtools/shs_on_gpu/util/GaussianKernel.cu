#include "GaussianKernel.hpp"

#include <cuda.h>

extern "C"
{

spGKernel GaussianKernel::makeKernel(float standardDeviation,
    std::string streamName,
    bool persistent)
{
    printf("===== LETS MAKE A KERNEL =)===\n");
    return spGKernel(new GaussianKernel(standardDeviation, streamName, persistent));
}

GaussianKernel::~GaussianKernel()
{
    if (d_kernel != nullptr && !mPersistent)
        cudaFree(d_kernel);
}

void GaussianKernel::copyKernelToGPU()
{
    int memsize = sizeof(float)*mKernelSize*mKernelSize;
    if (d_kernel != nullptr)
        cudaMalloc(&d_kernel, memsize);
    float* src = mp_IHkernel->getWriteBuffer();
    cudaMemcpy(d_kernel, src, memsize, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

GaussianKernel::GaussianKernel(float standardDeviation,
    std::string streamName,
    bool persistent)
    : mStandardDeviation(standardDeviation), mPersistent(persistent)
{
    // Determine the kernel size
    // => Include 4 stdDevs
    mKernelSize = (size_t) ceil(4*mStandardDeviation);
    if ((mKernelSize % 2) == 0)
        mKernelSize ++;  // The kernel size should be odd
    mKernelCenter = (int) floor(mKernelSize/2.);
    printf("Kernel size = %zu, kernel center @ %d\n", mKernelSize, mKernelCenter);
    
    // Generate the kernel image handler
    mp_IHkernel = SGR_ImageHandler<float>::newImageHandler(
        streamName.c_str(), mKernelSize, mKernelSize);
    mp_IHkernel->setPersistent(mPersistent);
    
    // Build the kernel
    float kernelSum = 0;
    for (size_t ix = 0; ix < mKernelSize; ix++)
        for (size_t iy = 0; iy < mKernelSize; iy++)
        {
            float x = ix-mKernelCenter;
            float y = iy-mKernelCenter;
            float val = exp(-(x*x+y*y)/(2*mStandardDeviation*mStandardDeviation));
            mp_IHkernel->write(val, ix, iy);
            kernelSum += val;
        }

    // Normalize the kernel to its energy
    for (size_t ix = 0; ix < mKernelSize; ix++)
        for (size_t iy = 0; iy < mKernelSize; iy++)
            mp_IHkernel->write(mp_IHkernel->read(ix, iy)/kernelSum, ix, iy);
    mp_IHkernel->updateWrittenImage();
}

}