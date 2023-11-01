#ifndef GAUSSIANKERNEL_HPP
#define GAUSSIANKERNEL_HPP

#include "ImageHandler2D.hpp"

#define spGKernel std::shared_ptr<GaussianKernel>

// A class for evaluating SHS images on a GPU
class GaussianKernel
{
public:
    static spGKernel makeKernel(
        float standardDeviation,
        std::string streamName,
        bool persistent);
    ~GaussianKernel();

    // Returns the standard deviation of the kernel
    float getStdDev() { return mStandardDeviation; }
    size_t getKernelSize() { return mKernelSize; }
    // Returns the image handler of the kernel
    spImHandler2D(float) getKernelIH() { return mp_IHkernel; }

    // Returns a pointer to a kernel copy on the device.
    // If no copy exists yet, the data is copied to the device.
    // ATTENTION: If persistent is set to true, this memory
    // will not be freed upon destruction!
    float* getPointerToDeviceCopy() { return mp_IHkernel->getGPUCopy(); }

private:
    float mStandardDeviation;
    size_t mKernelSize;
    int mKernelCenter;
    spImHandler2D(float) mp_IHkernel;
    bool mPersistent;

    float* d_kernel = nullptr;

    GaussianKernel(); // No publically available default DTor
    GaussianKernel(float standardDeviation,
        std::string streamName,
        bool persistent);
};

#endif // GAUSSIANKERNEL_HPP
