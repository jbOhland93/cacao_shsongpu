#include <cuda.h>
#include "SGE_GridLayout.hpp"

__global__ void evaluateSpots(
    uint16_t* h_imageData,
    float* d_darkData,
    int imW,
    SGE_GridLayout* d_GridLayout,
    uint16_t* d_windowCentersX,
    uint16_t* d_windowCentersY,
    float* d_kernel,
    int* d_convCoordsX,
    int* d_convCoordsY,
    float* d_debugImage,
    float* d_debugBuffer)
{    
// == Initialize local variables
    // Copy the grid layout to the register
    // But avoid calling a ctor or dtor
    char glBuf[sizeof(SGE_GridLayout)];
    SGE_GridLayout* GL = (SGE_GridLayout*)glBuf;
    *GL = *d_GridLayout;
    // Get the root of the current window
    int windowRootX = d_windowCentersX[blockIdx.x] - GL->mWindowSize/2;
    int windowRootY = d_windowCentersY[blockIdx.x] - GL->mWindowSize/2;

// == Chop up dynamic shared memory
    extern __shared__ float shm[];
    // Array for the image data of the block window
    float* imData = &shm[GL->mShmImDataOffset];
    // Array for the kernel values for the convolution
    float* kernel = &shm[GL->mShmKernelOffset];
    // Array for the X-coordinates of the convolution area
    int* convCoordsX = (int*)&shm[GL->mShmConvCoordsXOffset];
    // Array for the Y-coordinates of the convolution area
    int* convCoordsY = (int*)&shm[GL->mShmConvCoordsYOffset];
    // Array for buffering the intermediate results of the convolution
    float* convBuffer1 = &shm[GL->mShmConvBuf1Offset];
    // Array for buffering the intermediate results of the convolution
    float* convBuffer2 = &shm[GL->mShmConvBuf2Offset];
    // Array for the result of the local convolution
    float* convResults = &shm[GL->mShmConvResultOffset];
    
// == Stream data into shared memory
    if (threadIdx.x < GL->mNumWindowPx)
    {   // Stream pixel values to shm and do dark subtraction
        int imX = windowRootX + threadIdx.x % GL->mWindowSize;
        int imY = windowRootY + threadIdx.x / GL->mWindowSize;
        imData[threadIdx.x] =
            (float)h_imageData[imY*imW+imX]
            - d_darkData[imY*imW+imX];
    }
    else if (threadIdx.x < GL->mNumWindowPx + GL->mNumKernelPx)
    {   // Stream kernel values to shm
        int idx = threadIdx.x - GL->mNumWindowPx ;
        kernel[idx] = d_kernel[idx];
    }
    else if (threadIdx.x < GL->mNumWindowPx + GL->mNumKernelPx + GL->mNumCorrelPosPerAp)
    {   // Stream X-values of the convolution area to shm
        int idx = threadIdx.x - GL->mNumWindowPx - GL->mNumKernelPx;
        convCoordsX[idx] = d_convCoordsX[idx];
    }
    else if (threadIdx.x < GL->mNumWindowPx + GL->mNumKernelPx + 2*GL->mNumCorrelPosPerAp)
    {   // Stream Y-values of the convolution area to shm
        int idx = threadIdx.x - GL->mNumWindowPx - GL->mNumKernelPx - GL->mNumCorrelPosPerAp;
        convCoordsY[idx] = d_convCoordsY[idx];
    }
    __syncthreads();
// ## Sanity check: Feed streamed pixels into output image
/*    if (threadIdx.x < GL->mNumWindowPx)
    {
        int imX = windowRootX + threadIdx.x % GL->mWindowSize;
        int imY = windowRootY + threadIdx.x / GL->mWindowSize;
        d_debugImage[imY*imW+imX] = imData[threadIdx.x]-10;
    }*/


// == Do convolution for the selected locations
    // Each thread performs exactly one multiplication of a pixel and a kernel value
    // These values are stored in the convolution buffer and are added in the next step
    int convCoordIdx = threadIdx.x / GL->mNumKernelPx;
    int kernelIdx = threadIdx.x % GL->mNumKernelPx;
    bool convThread = convCoordIdx < GL->mNumCorrelPosPerAp;
    if (convThread)
    {
        int apX = convCoordsX[convCoordIdx] + kernelIdx % GL->mKernelSize - GL->mCorrelMargin;
        int apY = convCoordsY[convCoordIdx] + kernelIdx / GL->mKernelSize - GL->mCorrelMargin;
        convBuffer1[threadIdx.x] = kernel[kernelIdx] * imData[apY*GL->mWindowSize + apX];
    }
    __syncthreads();

    // Integrate over each kernel-patch of the convolution buffer to get the convolution result
    // Fields for buffer swapping
    float* srcBuffer;
    float* dstBuffer;
    bool bufferSelector = false;
    // Fields for array access
    int storageOffset = convCoordIdx * GL->mNumKernelPx;
    int reductionIndex = threadIdx.x - storageOffset;
    int indexA, indexB;
    // Fields for iteration tracking
    int sumCounter = GL->mNumKernelPx; // Number of elements to be added
    int activeThreads;

    while (sumCounter > 1)
    {
        // Swap source- and destination buffers
        srcBuffer = bufferSelector ? convBuffer2 : convBuffer1;
        dstBuffer = bufferSelector ? convBuffer1 : convBuffer2;
        bufferSelector = !bufferSelector;
        // Number of additions to be done in this iteration = number of active threads
        activeThreads = (sumCounter+1)/2;
        // Add the specified elements and write them to the destination buffer
        if (convThread && reductionIndex < activeThreads)
        {
            indexA = 2*reductionIndex;
            indexB = indexA+1;
            if (indexB < sumCounter)
                dstBuffer[threadIdx.x] =
                        srcBuffer[indexA + storageOffset]
                        + srcBuffer[indexB + storageOffset];
            else    // Odd array size? Just pass the last value to the next iteration
                dstBuffer[threadIdx.x] = srcBuffer[indexA + storageOffset];
                
        }
        // Update the number of elements to be added
        sumCounter = activeThreads;
        // Sync after each loop iteration as we rely on the results of all threads
        __syncthreads();
    }
    // Integration complete. Collect the results.
    if (reductionIndex == 0)
        convResults[convCoordIdx] = dstBuffer[threadIdx.x];
    __syncthreads();

    int offX = 5;
// ## Sanity check: Feed convoluted pixels into output image
/*    if (threadIdx.x < GL->mNumCorrelPosPerAp)
    {
        int convCoordX = convCoordsX[threadIdx.x];
        int convCoordY = convCoordsY[threadIdx.x];
        int imX = windowRootX + convCoordX;
        int imY = windowRootY + convCoordY;
        // For better visibility: Negate the ovolution result.
        d_debugImage[imY*imW+imX+offX] = -convResults[threadIdx.x];
        d_debugImage[imY*imW+imX-offX] = -convResults[threadIdx.x];
    }*/


// == Determine the spot center from the convolution result
    // There's not much to be parallelized here, so let's do it single threaded.
    if (threadIdx.x == 0)
    {
        int maxPosIdx = 0;
        float maxVal = convResults[maxPosIdx];
        float curVal = maxVal;
        for (int i = 0; i < GL->mNumCorrelPosPerAp; i++)
        {
            curVal = convResults[i];
            if (maxVal < curVal)
            {
                maxVal = curVal;
                maxPosIdx = i;
            }
            // Copy the colvoluted values to the shm array of image data for 2D access
            imData[convCoordsY[i]*GL->mWindowSize + convCoordsX[i]] = curVal;
        }
        int maxIdxX = convCoordsX[maxPosIdx];
        int maxIdxY = convCoordsY[maxPosIdx];
        int centerIndex = GL->mWindowSize/2 - 1;

        bool xInRange = (maxIdxX == centerIndex) || (maxIdxX == centerIndex+1);
        bool yInRange = (maxIdxY == centerIndex) || (maxIdxY == centerIndex+1);

        float spotPosXInWindow;
        float spotPosYInWindow;
        if (xInRange && yInRange)
        {   // Spot is still in tracking rectangle
            // Calculate the x-position of the spot
            float xNeg = imData[maxIdxY*GL->mWindowSize + maxIdxX-1];
            float xPos = imData[maxIdxY*GL->mWindowSize + maxIdxX+1];
            float fracX = (xNeg - xPos)/(xNeg + xPos - 2*maxVal) / 2;
            spotPosXInWindow = maxIdxX + fracX;
            
            // Calculate the y-position of the spot
            float yNeg = imData[(maxIdxY-1)*GL->mWindowSize + maxIdxX];
            float yPos = imData[(maxIdxY+1)*GL->mWindowSize + maxIdxX];
            float fracY = (yNeg - yPos)/(yNeg + yPos - 2*maxVal) / 2;
            spotPosYInWindow = maxIdxY + fracY;
        }
        else
        {   // Lost track of the spot
            // For now, just assume the spot did not wander too far and that the
            // max value of the convolution area points towards the spot center.
            // Therefore, setting the position of the max value as spot center,
            // The search window will drift towards the spot, hopefully catching it
            // in upcomong measurements.
            // If that is not stable enough, implement a different routine in the future.
            spotPosXInWindow = maxIdxX;
            spotPosYInWindow = maxIdxY;
        }
        float spotPositionX = windowRootX + spotPosXInWindow;
        float spotPositionY = windowRootY + spotPosYInWindow;

        //d_debugImage[((int)spotPositionY)*imW + (int)spotPositionX + offX] = 1000;
        /*d_debugImage[(maxIdxY + windowRootY)*imW + maxIdxX + windowRootX] = 1000;
        for (int i = 0; i < GL->mNumCorrelPosPerAp; i++)
        {
            d_debugImage[(convCoordsY[i] + windowRootY)*imW + convCoordsX[i] + windowRootX + 10] = 50*i;
        }
        d_debugImage[(convCoordsY[maxPosIdx] + windowRootY)*imW + windowRootX + convCoordsX[maxPosIdx] + 10] = 1000;
        */

        // If the spot drifted out of the center of the tracking rectangle,
        // update the window root positions accordingly.
        if (spotPosXInWindow < ((float)centerIndex))
            d_windowCentersX[blockIdx.x]--;
        else if (spotPosXInWindow > centerIndex+1.f)
            d_windowCentersX[blockIdx.x]++;

        if (spotPosYInWindow < ((float)centerIndex))
            d_windowCentersY[blockIdx.x]--;
        else if (spotPosYInWindow > centerIndex+1.f)
            d_windowCentersY[blockIdx.x]++;

        if (threadIdx.x == 0)
        {
            d_debugBuffer[blockIdx.x*6] = maxIdxX;
            d_debugBuffer[blockIdx.x*6+1] = maxIdxY;
            d_debugBuffer[blockIdx.x*6+2] = spotPosXInWindow;
            d_debugBuffer[blockIdx.x*6+3] = spotPosYInWindow;
            d_debugBuffer[blockIdx.x*6+4] = spotPositionX;
            d_debugBuffer[blockIdx.x*6+5] = spotPositionY;
            d_debugBuffer[blockIdx.x*6+6] = centerIndex;
            d_debugBuffer[blockIdx.x*6+7] = xInRange;
            d_debugBuffer[blockIdx.x*6+8] = yInRange;
            d_debugBuffer[blockIdx.x*6+9] = xInRange && yInRange;
        }
    }
}
