#ifndef SGE_GRIDLAYOUT_HPP
#define SGE_GRIDLAYOUT_HPP

#include "../util/CudaSGEutil.hpp"

class SGE_GridLayout
{
public:
    SGE_GridLayout(
        int device,
        int numSubapertures,
        int kernelSize,
        int warpsPerBlock);
    ~SGE_GridLayout();

    // GPU data
    int mDeviceID;              // Device ID
    int mNumCudaCores;          // Cuda cores on GPU
    int mWarpSize;              // GPU warp size

    // Grid dimensioning
    int mWarpsPerBlock;         // Warps per block
    int mBlockSize;             // Threads per block
    int mNumSubapertures;       // Total # of apertures
    int mNumBlocks;             // # of blocks in grid
    int mAperturesPerBlockBulk; // # of apertures to be evaluated per block
    int mAperturesPerBlockLast; // # of apertures to be evaluated in last block

    // Aperture data
    int mKernelSize;            // Edge size of the convolution kernel
    int mNumKernelPx;           // # of pixels in the kernel
    int mCorrelMargin;          // Margin in px between window edge and correl. area
    int mWindowSize;            // Edge size of considered window per subaperture
    int mNumWindowPx;           // # of pixels inside one aperture window
    int mStreamedPxPerThread;   // # of pixels that each thread has to stream
    int mCorrelCalcsPerThread;  // # of multipl. of Kernel- and Window pixels / thread
    int mNumCorrelPosPerAp = 12;// # of positions for which the correlation shall be calced

    // Shared memory organization
    int mShmSize;               // Size of the shared memory per block in bytes
    int mShmOffsetPixels;       // Offset of the pixel data in sizeof(float)
    int mShmApStridePixels;     // Stride width for one aperture in sizeof(float)
    int mShmOffsetKernel;       // Offset to the kernel data in sizeof(float)
    int mShmOffsetRootsX;       // Offset to the aperture rootsX in the image in sizeof(float)
    int mShmOffsetRootsY;       // Offset to the aperture rootsY in the image in sizeof(float)
    int mShmOffsetCCoordX;      // Offset to the correlation offsetX in sizeof(float)
    int mShmOffsetCCoordY;      // Offset to the correlation offsetY in sizeof(float)
    int mShmOffsetCorrelRes;    // Offst to the Correlation results in sizeof(float)
    int mShmApStrideCorrelRes;  // Stride width for one aperture in sizeof(float)

    SGE_GridLayout* getDeviceCopy() { return mpd_deviceCopy; }

    // Writes the index of the subaperture and location within
    // the subaperture to the given pointers
    CUDA_CALLABLE_MEMBER void getStreamStartIndicees(int blockIdx, int threadIdx, int* apIdxOut, int* apXout, int* apYout)
    {
        int firstApertureInBlock = blockIdx * mAperturesPerBlockBulk;
        int pxIndex = threadIdx * mStreamedPxPerThread;
        int pxApIndex = pxIndex % mNumWindowPx;
        *apXout = pxApIndex % mWindowSize;
        *apYout = pxApIndex / mWindowSize;
        *apIdxOut = firstApertureInBlock + pxIndex / mNumWindowPx;
    }

    // Allocates and writes coordinate offsets relative to the root
    // of a subaperture to the given locations.
    // The buffer is assumed to be the shared memory of a cuda block,
    // which is dimensioned and partitioned according to this class.
    CUDA_CALLABLE_MEMBER void gnrtCorrelOffsetsFrmRoots(float* shm, int index)
    {
        if (index >= 0 && index < mNumCorrelPosPerAp)
        {
            int* xOff = (int*)&shm[mShmOffsetCCoordX];
            int* yOff = (int*)&shm[mShmOffsetCCoordY];
            if (index < 2)
            {
                xOff[index] = mCorrelMargin + index + 1;
                yOff[index] = mCorrelMargin;
            }
            else if (index < 6)
            {
                xOff[index] = mCorrelMargin + index - 2;
                yOff[index] = mCorrelMargin + 1;
            }
            else if (index < 10)
            {
                xOff[index] = mCorrelMargin + index - 6;
                yOff[index] = mCorrelMargin + 2;
            }
            else if (index < 12)
            {
                xOff[index] = mCorrelMargin + index - 9;
                yOff[index] = mCorrelMargin + 3;
            }
        }
    }
    void gnrtCorrelOffsetsFrmRootsHost(int* xOff, int* yOff)
    {
        for (int index = 0; index < mNumCorrelPosPerAp; index++)
        {
            if (index < 2)
            {
                xOff[index] = mCorrelMargin + index + 1;
                yOff[index] = mCorrelMargin;
            }
            else if (index < 6)
            {
                xOff[index] = mCorrelMargin + index - 2;
                yOff[index] = mCorrelMargin + 1;
            }
            else if (index < 10)
            {
                xOff[index] = mCorrelMargin + index - 6;
                yOff[index] = mCorrelMargin + 2;
            }
            else if (index < 12)
            {
                xOff[index] = mCorrelMargin + index - 9;
                yOff[index] = mCorrelMargin + 3;
            }
        }
    }

private:
    SGE_GridLayout(); // No publically available default ctor.
    int getCudaCoresPerSM(int major, int minor);
    void printReport();
    void copyToGPU();

    SGE_GridLayout* mpd_deviceCopy;
};

#endif // SGE_GRIDLAYOUT_HPP