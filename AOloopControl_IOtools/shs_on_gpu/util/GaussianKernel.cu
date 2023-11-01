#include "GaussianKernel.hpp"

#include <cuda.h>

extern "C"
{

spGKernel GaussianKernel::makeKernel(float standardDeviation,
    std::string streamName,
    bool persistent)
{
    return spGKernel(new GaussianKernel(standardDeviation, streamName, persistent));
}

GaussianKernel::~GaussianKernel()
{
    if (d_kernel != nullptr && !mPersistent)
        cudaFree(d_kernel);
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
    printf("Kernel: stddev = %.3f, size = %zu, kernel center @ %d\n",
        mStandardDeviation,
        mKernelSize,
        mKernelCenter);
    
    // Generate the kernel image handler
    mp_IHkernel = ImageHandler2D<float>::newImageHandler2D(
        streamName.c_str(), mKernelSize, mKernelSize);
    mp_IHkernel->setPersistent(mPersistent);
    
    // Build the kernel
    float kernelSum = 0;
    for (int ix = 0; ix < (int) mKernelSize; ix++)
        for (int iy = 0; iy < (int) mKernelSize; iy++)
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