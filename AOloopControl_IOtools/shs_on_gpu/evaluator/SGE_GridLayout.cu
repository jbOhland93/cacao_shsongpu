#include "SGE_GridLayout.hpp"

#include <stdio.h>
#include <cuda.h>
#include <string>
#include <stdexcept>

SGE_GridLayout::SGE_GridLayout(
    int device,
    int numSubapertures,
    int kernelSize,
    int warpsPerBlock)
    : mNumSubapertures(numSubapertures),
      mKernelSize(kernelSize),
      mNumKernelPx(kernelSize*kernelSize),
      mWarpsPerBlock(warpsPerBlock),
      mDeviceID(device)
{
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    printCE(err)
    int MPcount = prop.multiProcessorCount;
    mNumCudaCores = getCudaCoresPerSM(prop.major, prop.minor) * MPcount;
    mWarpSize = prop.warpSize;

    // Calc grid dimensioning
    mBlockSize = mWarpSize * warpsPerBlock;
    float coresPerAperture = mNumCudaCores / (float) numSubapertures;
    float blocksPerAperture = coresPerAperture / mBlockSize;
    mAperturesPerBlockBulk = ceil(1/blocksPerAperture);
    mNumBlocks = ceil(numSubapertures / (float) mAperturesPerBlockBulk);
    mAperturesPerBlockLast = mNumSubapertures - (mNumBlocks-1)*mAperturesPerBlockBulk;

    // Calc aperture dimensioning
    // We need an area of 4x4 pixels after correlation.
    // To perform this correlation, the kernel will overlap on each side.
    // 2*correlMargin accounts for this overlap.
    mCorrelMargin = (kernelSize - 1) / 2;
    mWindowSize = 4 + mCorrelMargin * 2;
    mNumWindowPx = mWindowSize * mWindowSize;
    int pixelsStreamedPerBlock = mNumWindowPx * mAperturesPerBlockBulk;
    mStreamedPxPerThread = ceil(pixelsStreamedPerBlock / (float) mBlockSize);
    int pxCorrelPerBlock = mNumCorrelPosPerAp * mAperturesPerBlockBulk;
    mCorrelCalcsPerThread = ceil(mKernelSize*mKernelSize * pxCorrelPerBlock / (float) mBlockSize);

    // Set shared memory organization
    mShmSize = 0;
    
    // Memory for pixel storage
    mShmOffsetPixels = 0;
    mShmApStridePixels = mNumWindowPx;
    int memsizePixels = mShmApStridePixels * mAperturesPerBlockBulk;
    mShmSize += memsizePixels * sizeof(float);
    // Memory for kernel storage
    mShmOffsetKernel = mShmOffsetPixels + memsizePixels;
    int memsizeKernel = mKernelSize * mKernelSize;
    mShmSize += memsizeKernel * sizeof(float);
    // Memory for the aperture window roots
    mShmOffsetRootsX = mShmOffsetKernel + memsizeKernel;
    int memsizeRoots = mNumSubapertures;
    mShmSize += memsizeRoots * sizeof(int); // sizeof(int) == sizeof(float)
    mShmOffsetRootsY = mShmOffsetRootsX + memsizeRoots;
    mShmSize += memsizeRoots * sizeof(int); // sizeof(int) == sizeof(float)
    // Memory for the correlation target coordinates;
    mShmOffsetCCoordX = mShmOffsetRootsY + memsizeRoots;
    int memsizeCCoord = mNumCorrelPosPerAp;
    mShmSize += memsizeCCoord * sizeof(int);
    mShmOffsetCCoordY = mShmOffsetCCoordX + memsizeCCoord;
    mShmSize += memsizeCCoord * sizeof(int);
    // Memory for correlation result storage
    mShmOffsetCorrelRes = mShmOffsetCCoordY + memsizeCCoord;
    mShmApStrideCorrelRes = mNumCorrelPosPerAp;
    int memsizeCorrelResult = mShmApStrideCorrelRes * mAperturesPerBlockBulk;
    mShmSize += memsizeCorrelResult * sizeof(float);

    printReport();

    // Copy this object to the GPU
    cudaMalloc(&mpd_deviceCopy, sizeof(SGE_GridLayout));
    cudaMemcpy(mpd_deviceCopy, this, sizeof(SGE_GridLayout), cudaMemcpyHostToDevice);
}

SGE_GridLayout::~SGE_GridLayout()
{
    // Delete the device copy from GPU memory
    cudaFree(mpd_deviceCopy);
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
    printf("- Max. apertures per block:\t\t%d\n", mAperturesPerBlockBulk);
    printf("- Streamed px. per thread:\t\t%d\n", mStreamedPxPerThread);
    printf("- Correl. mul. OPs per thread:\t\t%d\n", mCorrelCalcsPerThread);

    printf("\nGrid performance:\n");
    printf("- Core usage:\t\t\t\t%d (%.1f)\n",
        mNumBlocks*mBlockSize, mNumBlocks*mBlockSize/(float)mNumCudaCores*100);

    int streamingCoresInBulk = (mNumBlocks-1)*ceil(mNumWindowPx*mAperturesPerBlockBulk / (float) mStreamedPxPerThread);
    int streamingCoresInLast = ceil(mNumWindowPx*mAperturesPerBlockLast / (float) mStreamedPxPerThread);
    int totalStreamingCores = streamingCoresInBulk + streamingCoresInLast;
    printf("- Cores participating in streaming: \t%d (%.1f)\n",
        totalStreamingCores, totalStreamingCores/(float)mNumCudaCores*100);

    int correlCoresInBulk = (mNumBlocks-1)*ceil(mNumCorrelPosPerAp*mKernelSize*mKernelSize*mAperturesPerBlockBulk / (float) mCorrelCalcsPerThread);
    int correlCoresInLast = ceil(mNumCorrelPosPerAp*mKernelSize*mKernelSize*mAperturesPerBlockLast / (float) mCorrelCalcsPerThread);
    int totalCorrelCores = correlCoresInBulk + correlCoresInLast;
    printf("- Cores participating in correlation: \t%d (%.1f)\n",
        totalCorrelCores, totalCorrelCores/(float)mNumCudaCores*100);

    printf("\n=== SGE_GridLayout end. ===\n\n");
}

