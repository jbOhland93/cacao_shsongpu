#include <cuda.h>
#include "SGE_GridLayout.hpp"

__device__ void getCorrellationIndicees(
    int threadIndex,
    int numCorrelOpsPerThread,
    int kernelSize,
    int numCorrelCoordsPerAp,
    int* apertureIndexOut,
    int* coordinateIndexOut,
    int* kernelIndexYOut,
    int* kernelIndexXOut)
{
    int opIdx = threadIndex * numCorrelOpsPerThread;
    
    int opsPerCoord = kernelSize*kernelSize;
    int opsPerAp = numCorrelCoordsPerAp*opsPerCoord;

    *apertureIndexOut = opIdx / opsPerAp;
    opIdx -= *apertureIndexOut * opsPerAp;
    *coordinateIndexOut = opIdx / opsPerCoord;
    opIdx -= *coordinateIndexOut * opsPerCoord;
    *kernelIndexYOut = opIdx / kernelSize;
    *kernelIndexXOut = opIdx - *kernelIndexYOut * kernelSize;
}

__global__ void evaluateSpot(
    uint16_t* h_imageData,
    float* d_darkData,
    float* d_kernel,
    int imW,
    uint16_t* d_roiPosX,
    uint16_t* d_roiPosY,
    SGE_GridLayout* d_gridLayout,
    float* d_intensity,
    float* d_convImage)
{
    //if (blockIdx.x < d_gridLayout->mNumSubapertures)
    //{
        // Initialize shared memory
        extern __shared__ float shm[];
        __shared__ char* gridLayoutSpace[sizeof(SGE_GridLayout)];

        // Copy the grid layout
        if (threadIdx.x == 0)
            *(SGE_GridLayout*)gridLayoutSpace = *d_gridLayout;
        __syncthreads();

        // Chop up the shared memory
        SGE_GridLayout* gridLayout = (SGE_GridLayout*)gridLayoutSpace;
        float* imageData = &shm[gridLayout->mShmOffsetPixels];
        float* kernel = &shm[gridLayout->mShmOffsetKernel];
        float* correlationResult = &shm[gridLayout->mShmOffsetCorrelRes];
        int* apRootsX = (int*)&shm[gridLayout->mShmOffsetRootsX];
        int* apRootsY = (int*)&shm[gridLayout->mShmOffsetRootsY];
        int* correlCoordX = (int*)&shm[gridLayout->mShmOffsetCCoordX];
        int* correlCoordY = (int*)&shm[gridLayout->mShmOffsetCCoordY];

        // Fill in static data to shared memory
        int numSubapertures = gridLayout->mNumSubapertures;
        int kernelSize = gridLayout->mKernelSize;
        int apPerBlockBulk = gridLayout->mAperturesPerBlockBulk;
        if (threadIdx.x < gridLayout->mNumCorrelPosPerAp)
            gridLayout->gnrtCorrelOffsetsFrmRoots(shm, threadIdx.x);
        else if (threadIdx.x == blockDim.x-1)
            for (int i = 0; i < numSubapertures; i++)
                apRootsX[i] = d_roiPosX[i];
        else if (threadIdx.x == blockDim.x-2)
            for (int i = 0; i < numSubapertures; i++)
                apRootsY[i] = d_roiPosY[i];
        else if (threadIdx.x == blockDim.x-3)
            for (int i = 0; i < kernelSize*kernelSize; i++)
                kernel[i] = d_kernel[i];

        // Get aperture indicees
        int apIndex, apX, apY;
        gridLayout->getStreamStartIndicees(blockIdx.x, threadIdx.x, &apIndex, &apX, &apY);
        int blockApIndex = apIndex % apPerBlockBulk;
        __syncthreads();

        // Stream pixels
        int winSize;
        int numAperturePixels;
        int pixelIndex;
        int toStream;
        int rootX;
        int rootY;
        if (apIndex > numSubapertures)
            toStream = 0;
        else
        {
            winSize = gridLayout->mWindowSize;
            numAperturePixels = gridLayout->mNumWindowPx;
            pixelIndex = blockApIndex*numAperturePixels + apY*winSize + apX;
            toStream = gridLayout->mStreamedPxPerThread;
            rootX = apRootsX[apIndex];
            rootY = apRootsY[apIndex];
        }
        while (toStream > 0)
        {
            int imX = rootX + apX;
            int imY = rootY + apY;
            imageData[pixelIndex] =
                (float) h_imageData[imY*imW + imX]
                - d_darkData[imY*imW + imX];
            toStream--;
            // If there are still pixels to stream, update indices
            if (toStream > 0)
            {
                pixelIndex++;
                apX++;
                if (apX == winSize)
                {
                    apX = 0;
                    apY++;
                    if (apY == winSize)
                    {
                        apY = 0;
                        apIndex++;
                        if (apIndex > numSubapertures)
                            toStream = 0;
                        else
                        {
                            rootX = apRootsX[apIndex];
                            rootY = apRootsY[apIndex];
                        }
                    }
                }
            }
        }

        // Fetch data for the convolution before syncing
        int firstApertureInBlock = blockIdx.x * apPerBlockBulk;
        int correlOpsPerThread = gridLayout->mCorrelCalcsPerThread;
        int correlCoordsPerAprt = gridLayout->mNumCorrelPosPerAp;

        __syncthreads();
        
        // Do tile processing
        int blckApertIdx;
        int coordIdx;
        int kIdxY;
        int kIdxX;
        getCorrellationIndicees(
            threadIdx.x,
            correlOpsPerThread,
            kernelSize,
            correlCoordsPerAprt,
            &blckApertIdx,
            &coordIdx,
            &kIdxY,
            &kIdxX);


        int apertIdx = blockIdx.x*apPerBlockBulk + blckApertIdx;
        if (apertIdx < numSubapertures)
        {
            int correlMargin = gridLayout->mCorrelMargin;

            int rootX = apRootsX[apertIdx];
            int rootY = apRootsY[apertIdx];
            int apCoordX = correlCoordX[coordIdx];
            int apCoordY = correlCoordY[coordIdx];



// Set to 0 if needed
        if (threadIdx.x == 0)
            for (int blkAp = 0; blkAp < apPerBlockBulk; blkAp++)
                for (int coordIdx = 0; coordIdx < correlCoordsPerAprt; coordIdx++)
                    {
                        int apIdx = blkAp + blockIdx.x * apPerBlockBulk;
                        if (apIdx < numSubapertures)
                            correlationResult[blkAp*correlCoordsPerAprt + coordIdx] = 0;
                    }




            // Do convlution!
            int toConvolve = correlOpsPerThread;
            float sum = 0.f;
            while (toConvolve > 0)
            {
                sum += 1;
                    //kernel[kIdxY*kernelSize+kIdxX] *
                    //abs(imageData[blckApertIdx*numAperturePixels + apCoordY*winSize + apCoordX]);
                correlationResult[blckApertIdx*correlCoordsPerAprt + coordIdx] = 1;
                toConvolve--;
                // Done?
                if (toConvolve == 0)
                {
                    // Add sum to correl results
                    //correlationResult[blckApertIdx*correlCoordsPerAprt + coordIdx] = 1;//+= sum;
                }
                else
                {
                    kIdxX++;
                    // Kernel row completed?
                    if (kIdxX == kernelSize)
                    {
                        kIdxX = 0;
                        kIdxY++;
                        // Kernel completed?
                        if (kIdxY == kernelSize)
                        {
                            // Add sum to the correlation result
                            //correlationResult[blckApertIdx*correlCoordsPerAprt + coordIdx] = 1;//+= sum;
                            sum = 0;
                            kIdxY = 0;
                            coordIdx++;
                            if (coordIdx == correlCoordsPerAprt)
                            {
                                coordIdx = 0;
                                apertIdx++;
                                blckApertIdx++;
                                if (apertIdx == numSubapertures && blckApertIdx == apPerBlockBulk)
                                    break;
                                else
                                {
                                    rootX = apRootsX[apertIdx];
                                    rootY = apRootsY[apertIdx];
                                }
                            }
                            apCoordX = correlCoordX[coordIdx];
                            apCoordY = correlCoordY[coordIdx];
                        }
                    }
                }
            }
        }
        __syncthreads();
        // Copy convoluted image to array for testing purposes
        if (threadIdx.x == 0)
        {
            for (int x = 0; x < 480; x++)
                for (int y = 0; y < 424; y++)
                    d_convImage[y*imW + x] = 0;

            for (int blkAp = 0; blkAp < apPerBlockBulk; blkAp++)
                for (int coordIdx = 0; coordIdx < correlCoordsPerAprt; coordIdx++)
                    {
                        int apIdx = blkAp + blockIdx.x * apPerBlockBulk;
                        if (apIdx < numSubapertures)
                        {
                            int rootX = apRootsX[apIdx];
                            int rootY = apRootsY[apIdx];
                            int apX = correlCoordX[coordIdx];
                            int apY = correlCoordY[coordIdx];
                            d_convImage[(rootY+apY)*imW + rootX + apX]
                                = correlationResult[blkAp*correlCoordsPerAprt + coordIdx];
                        }
                    }
        }

        int inspectedIdx = 107;
        if (threadIdx.x == inspectedIdx && blockIdx.x == 79 )
        {
            //float sum = 0.;
            //int start = gridLayout->mNumWindowPx * (threadIdx.x);
            //int end = gridLayout->mNumWindowPx * (threadIdx.x+1);
            //for (int x = start; x < end; x++)
            //    sum += imageData[x];
            d_intensity[0] = threadIdx.x;
            d_intensity[1] = apertIdx;
            d_intensity[2] = coordIdx;
            d_intensity[3] = kIdxX;
            d_intensity[4] = kIdxY;
            d_intensity[5] = correlOpsPerThread;
            d_intensity[6] = correlCoordsPerAprt;
        }
}


__global__ void evaluateSingleSpot(
    uint16_t* h_imageData,
    float* d_darkData,
    float* d_outputImage,
    int imW,
    int windowRootX,
    int windowRootY,
    int windowSize,
    float* d_kernel,
    int kernelSize,
    int* d_convCoordsX,
    int* d_convCoordsY,
    int numConvCoords)
{
    extern __shared__ float shm[];
// Chop up dynamic shared memory
    int numPxInWindow = windowSize*windowSize;
    int numPxInKernel = kernelSize*kernelSize;
    float* imData = shm;
    float* kernel = &shm[numPxInWindow];
    int* convCoordsX = (int*)&shm[numPxInWindow + numPxInKernel];
    int* convCoordsY = (int*)&shm[numPxInWindow + numPxInKernel + numConvCoords];

// == Stream data into shared memory
    if (threadIdx.x < numPxInWindow)
    {
        int imX = windowRootX + threadIdx.x % windowSize;
        int imY = windowRootY + threadIdx.x / windowSize;
        imData[threadIdx.x] =
            (float)h_imageData[imY*imW+imX]
            - d_darkData[imY*imW+imX];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel)
    {
        int idx = threadIdx.x - numPxInWindow ;
        kernel[idx] = d_kernel[idx];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel + numConvCoords)
    {
        int idx = threadIdx.x - numPxInWindow - numPxInWindow;
        convCoordsX[idx] = d_convCoordsX[idx];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel + 2*numConvCoords)
    {
        int idx = threadIdx.x - numPxInWindow - numPxInWindow - numConvCoords;
        convCoordsY[idx] = d_convCoordsX[idx];
    }

// == Sanity check: Feed streamed pixels into output image
    if (threadIdx.x < numPxInWindow)
    {
        int imX = windowRootX + threadIdx.x % windowSize;
        int imY = windowRootY + threadIdx.x / windowSize;
        d_outputImage[imY*imW+imX] = imData[threadIdx.x]+10;
    }
}

__global__ void evaluateSpots(
    uint16_t* h_imageData,
    float* d_darkData,
    float* d_outputImage,
    int imW,
    uint16_t* windowRootsX,
    uint16_t* windowRootsY,
    int windowSize,
    float* d_kernel,
    int kernelSize,
    int* d_convCoordsX,
    int* d_convCoordsY,
    int numConvCoords,
    int correlMargin,
    float* d_debugBuffer)
{
    extern __shared__ float shm[];
// Chop up dynamic shared memory
    int numPxInWindow = windowSize*windowSize;
    int numPxInKernel = kernelSize*kernelSize;
    float* imData = shm;
    float* kernel = &shm[numPxInWindow];
    int* convCoordsX = (int*)&shm[numPxInWindow + numPxInKernel];
    int* convCoordsY = (int*)&shm[numPxInWindow + numPxInKernel + numConvCoords];
    float* convResults = &shm[numPxInWindow + numPxInKernel + 2*numConvCoords];

    int windowRootX = windowRootsX[blockIdx.x];
    int windowRootY = windowRootsY[blockIdx.x];
// == Stream data into shared memory
    if (threadIdx.x < numPxInWindow)
    {
        int imX = windowRootX + threadIdx.x % windowSize;
        int imY = windowRootY + threadIdx.x / windowSize;
        imData[threadIdx.x] =
            (float)h_imageData[imY*imW+imX]
            - d_darkData[imY*imW+imX];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel)
    {
        int idx = threadIdx.x - numPxInWindow ;
        kernel[idx] = d_kernel[idx];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel + numConvCoords)
    {
        int idx = threadIdx.x - numPxInWindow - numPxInKernel;
        convCoordsX[idx] = d_convCoordsX[idx];
    }
    else if (threadIdx.x < numPxInWindow + numPxInKernel + 2*numConvCoords)
    {
        int idx = threadIdx.x - numPxInWindow - numPxInKernel - numConvCoords;
        convCoordsY[idx] = d_convCoordsY[idx];
    }
    __syncthreads();

// == Sanity check: Feed streamed pixels into output image
/*    if (threadIdx.x < numPxInWindow)
    {
        int imX = windowRootX + threadIdx.x % windowSize;
        int imY = windowRootY + threadIdx.x / windowSize;
        d_outputImage[imY*imW+imX] = imData[threadIdx.x]-10;
    }
*/
    int convCoordIdx = threadIdx.x / numPxInKernel;
    int kernelIdx = threadIdx.x % numPxInKernel;
    if (convCoordIdx < numConvCoords)
    {
        int apX = convCoordsX[convCoordIdx] + kernelIdx % kernelSize - correlMargin;
        int apY = convCoordsY[convCoordIdx] + kernelIdx / kernelSize - correlMargin;
        convResults[threadIdx.x] = kernel[kernelIdx] * imData[apY*windowSize + apX];
    }
// == Sanity check: Feed convoluted pixels into output image
    if (threadIdx.x < numConvCoords)
    {
        int convCoordIdx = threadIdx.x;
        float convSum = 0.f;
        for (int i = 0; i < numPxInKernel; i++)
        {
            convSum += convResults[convCoordIdx*numPxInKernel + i];
        }
        int convCoordX = convCoordsX[convCoordIdx];
        int convCoordY = convCoordsY[convCoordIdx];
        int imX = windowRootX + convCoordX;
        int imY = windowRootY + convCoordY;

        int apX = convCoordsX[convCoordIdx] + kernelIdx % kernelSize - correlMargin;
        int apY = convCoordsY[convCoordIdx] + kernelIdx / kernelSize - correlMargin;

        d_outputImage[imY*imW+imX] = convSum;
    }
}
