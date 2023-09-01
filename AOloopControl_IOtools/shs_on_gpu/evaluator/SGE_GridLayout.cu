#include "SGE_GridLayout.hpp"

#include <stdio.h>
#include <cuda.h>
#include <string>
#include <stdexcept>

spGridLayout SGE_GridLayout::makeGridLayout(int device, spRefManager refManager)
{
    return spGridLayout(new SGE_GridLayout(device, refManager));
}

SGE_GridLayout::~SGE_GridLayout()
{
    // Delete the device copy from device memory
    cudaFree(mpd_deviceCopy);

    // Delete the correlation offsets from device memory
    cudaFree(mp_d_CorrelationOffsetsX);
    cudaFree(mp_d_CorrelationOffsetsY);
}

SGE_GridLayout::SGE_GridLayout(int device, spRefManager refManager)
    : mNumSubapertures(refManager->getNumSpots()),
      mKernelSize(refManager->getKernelSize()),
      mNumKernelPx(refManager->getKernelSize()*refManager->getKernelSize()),
      mDeviceID(device)
{
    // Set up grid layout based on the GPU properies
    getGPUproperties();
    calcGridProperties();
    setSHMlayout();

    // Write the correlation offsets to the GPU
    writeCorrelOffsetsFromWindowRootToDevice();

    // Copy this object to the GPU
    cudaMalloc(&mpd_deviceCopy, sizeof(SGE_GridLayout));
    cudaMemcpy(mpd_deviceCopy, this, sizeof(SGE_GridLayout), cudaMemcpyHostToDevice);

    // Finally, print a report about the grid layout.
    printReport();
}

void SGE_GridLayout::getGPUproperties()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, mDeviceID);
    printCE(err)
    int MPcount = prop.multiProcessorCount;
    mNumCudaCores = getCudaCoresPerSM(prop.major, prop.minor) * MPcount;
    mWarpSize = prop.warpSize;
}

int SGE_GridLayout::getCudaCoresPerSM(int major, int minor)
{
    std::string errMsg("GridSizeCalculator::getCudaCoresPerSM: ");
    switch (major)
    {
        case 2:
            if (minor == 0)
                return 32;
            else if (minor == 1)
                return 48;
        case 3:
            if (minor == 0 || minor == 5 || minor == 7)
                return 192;
        case 5:
            if (minor == 0 || minor == 2)
                return 128;
        case 6:
            if (minor == 0)
                return 64;
            else if (minor == 1)
                return 128;
        case 7:
            if (minor == 0 || minor == 5)
                return 64;
        case 8:
            if (minor == 0)
                return 64;
            if (minor == 6 || minor == 9)
                return 128;
        case 9:
            if (minor == 0)
                return 128;
        default:
            errMsg.append("no core count entry found for SM ");
            errMsg.append(std::to_string(major));
            errMsg.append(".");
            errMsg.append(std::to_string(minor));
            errMsg.append(". Please add an entry.\n");
            printf("%s", errMsg.c_str());
            throw std::runtime_error(errMsg);
    }
}

void SGE_GridLayout::calcGridProperties()
{
    // Calc grid dimensioning
    mBlockSize = mNumKernelPx;
    mBlockSize *= mNumCorrelPosPerAp;
    mWarpsPerBlock = mBlockSize/mWarpSize + 1;
    mBlockSize = mWarpsPerBlock * mWarpSize;
    mNumBlocks = mNumSubapertures;

    // Calc aperture dimensioning
    // We need an area of 4x4 pixels after correlation.
    // To perform this correlation, the kernel will overlap on each side.
    // 2*correlMargin accounts for this overlap.
    mCorrelMargin = (mKernelSize - 1) / 2;
    mWindowSize = 4 + mCorrelMargin * 2;
    mNumWindowPx = mWindowSize * mWindowSize;
}

void SGE_GridLayout::setSHMlayout()
{
    
    mShmSize = 0;
    int curOffset = 0;

    #define setMemoryBlock(memName, memSize) \
        memName ## Offset = curOffset; \
        memName ## Size = memSize; \
        mShmSize += memSize; \
        curOffset += memSize;

    // Holding window pixels in shm
    setMemoryBlock(mShmImData, mNumWindowPx);
    // Holding kernel pixels in shm
    setMemoryBlock(mShmKernel, mNumKernelPx);
    // Holding the convolution coordinates
    setMemoryBlock(mShmConvCoordsX, mNumCorrelPosPerAp);
    setMemoryBlock(mShmConvCoordsY, mNumCorrelPosPerAp);
    // Buffer for the calculation of the convulution
    setMemoryBlock(mShmConvBuf1, mNumCorrelPosPerAp*mNumKernelPx);
    setMemoryBlock(mShmConvBuf2, mNumCorrelPosPerAp*mNumKernelPx);
    // Holding the convolution results
    setMemoryBlock(mShmConvResult, mNumCorrelPosPerAp);

    // Convert the size to bytes
    mShmSize *= sizeof(float); // Note that sizeof(float) == sizeof(int)
}

void SGE_GridLayout::writeCorrelOffsetsFromWindowRootToDevice()
{   
    int xOff[SGE_GridLayout::getNumConvolutionsPerAp()];
    int yOff[SGE_GridLayout::getNumConvolutionsPerAp()];

    for(int i = 0;
        i < SGE_GridLayout::getNumConvolutionsPerAp();
        i++)
    {
        if (i < 2)
        {
            xOff[i] = mCorrelMargin + i + 1;
            yOff[i] = mCorrelMargin;
        }
        else if (i < 6)
        {
            xOff[i] = mCorrelMargin + i - 2;
            yOff[i] = mCorrelMargin + 1;
        }
        else if (i < 10)
        {
            xOff[i] = mCorrelMargin + i - 6;
            yOff[i] = mCorrelMargin + 2;
        }
        else if (i < 12)
        {
            xOff[i] = mCorrelMargin + i - 9;
            yOff[i] = mCorrelMargin + 3;
        }
    }

    // Copy the values to the device
    int bufsize = SGE_GridLayout::getNumConvolutionsPerAp()*sizeof(int);
    cudaMalloc(&mp_d_CorrelationOffsetsX, bufsize);
    cudaMalloc(&mp_d_CorrelationOffsetsY, bufsize);
    cudaMemcpy(mp_d_CorrelationOffsetsX, xOff, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpy(mp_d_CorrelationOffsetsY, yOff, bufsize, cudaMemcpyHostToDevice);
}

void SGE_GridLayout::printReport()
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, mDeviceID);
    printCE(err)

    printf("\n===== SGE_GridLayout: =====\n");

    printf("\nGPU properties:\n");
    printf("- Device ID:\t\t\t\t%d\n", mDeviceID);
    printf("- Name:\t\t\t\t\t%s\n", prop.name);
    printf("- Warp size:\t\t\t\t%d\n", mWarpSize);
    printf("- # of cores:\t\t\t\t%d\n", mNumCudaCores);

    printf("\nProblem metrics:\n");
    printf("- # of subapertures:\t\t\t%d\n", mNumSubapertures);
    printf("- Subaperture window size:\t\t%d (= %d pixels)\n", mWindowSize, mNumWindowPx);
    printf("- Kernel size:\t\t\t\t%d (= %d pixels)\n", mKernelSize, mKernelSize*mKernelSize);
    printf("- # of correlation locations:\t\t%d\n", mNumSubapertures*mNumCorrelPosPerAp);
    printf("- # of correl. mul. OPs:\t\t%d\n", mNumSubapertures*mNumCorrelPosPerAp*mKernelSize*mKernelSize);

    printf("\nGrid dimensioning:\n");
    printf("- Warps per block:\t\t\t%d\n", mWarpsPerBlock);
    printf("- Block size:\t\t\t\t%d\n", mBlockSize);
    printf("- # of blocks:\t\t\t\t%d\n", mNumBlocks);
    printf("- Core usage:\t\t\t\t%d (%.1f%%)\n",
        mNumBlocks*mBlockSize, mNumBlocks*mBlockSize/(float)mNumCudaCores*100);

    printf("\n=== SGE_GridLayout end. ===\n\n");
}

