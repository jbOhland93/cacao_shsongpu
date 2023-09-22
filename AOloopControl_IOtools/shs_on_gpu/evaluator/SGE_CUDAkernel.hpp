#include <cuda.h>
#include "SGE_GridLayout.hpp"


// === VISUAL DEBUG FLAGS ===
// Write output to the debug image array for visual inspection.
// Remember to define ENABLE_DEBUG_IMAGE in SGE_Evaluator.cu,
// otherwise it's a nullptr.
// Note: If you define more than one flag, the output may become messy.
// ===================

// visual inspection of the streaming process
//#define DEBUG_SHOW_STREAMED_PX

// visual inspection of the local convolution
//#define DEBUG_SHOW_LOCAL_CONVOLUTION

// visual inspection of the peak detection coordinates
//#define DEBUG_SHOW_MAX_PIXEL_COORDS

// === DEBUG FLAGS ===
// Provide commandline output for debugging purpose.
// Remember to define ENABLE_DEBUG_ARRAY in SGE_Evaluator.cu,
// otherwise it's a  nullptr.
// ===================
//#define DEBUG_ARRAY_OUTPUT

__global__ void evaluateSpots(
    uint16_t* h_imageData,
    float* d_darkData,
    int imW,
    SGE_GridLayout* d_GridLayout,
    uint16_t* d_windowCentersX,
    uint16_t* d_windowCentersY,
    float* d_kernel,
    float* d_refX,
    float* d_refY,
    float shift2gradConst,
    float outOfRangeDistance,
    float* d_gradOut,
    float* d_intensityOut,
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
    // The reference positions
    float refX = d_refX[blockIdx.x];
    float refY = d_refY[blockIdx.x];
    // Flag for judging if the spot is tracked well.
    // If this is set to false during the evaluation, the searching
    // rectangle will be reset to its initial position.
    bool measurementValid = true;

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
        convCoordsX[idx] = GL->mp_d_CorrelationOffsetsX[idx];
    }
    else if (threadIdx.x < GL->mNumWindowPx + GL->mNumKernelPx + 2*GL->mNumCorrelPosPerAp)
    {   // Stream Y-values of the convolution area to shm
        int idx = threadIdx.x - GL->mNumWindowPx - GL->mNumKernelPx - GL->mNumCorrelPosPerAp;
        convCoordsY[idx] = GL->mp_d_CorrelationOffsetsY[idx];
    }
    __syncthreads();

#ifdef DEBUG_SHOW_STREAMED_PX
// ## Sanity check: Feed streamed pixels back into output image
// ## using an offset for visual inspection of the streaming process
    if (threadIdx.x < GL->mNumWindowPx)
    {
        int imX = windowRootX + threadIdx.x % GL->mWindowSize;
        int imY = windowRootY + threadIdx.x / GL->mWindowSize;
        d_debugImage[imY*imW+imX] = imData[threadIdx.x]-100;
    }
#endif

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

#ifdef DEBUG_SHOW_LOCAL_CONVOLUTION
// ## Sanity check: Feed convoluted pixels into output image
// ## to visually inspect the local convolution
    int offX = 5; // Lateral offset in x-direction for comparison with original spot
    if (threadIdx.x < GL->mNumCorrelPosPerAp)
    {
        int convCoordX = convCoordsX[threadIdx.x];
        int convCoordY = convCoordsY[threadIdx.x];
        int imX = windowRootX + convCoordX;
        int imY = windowRootY + convCoordY;
        d_debugImage[imY*imW+imX+offX] = convResults[threadIdx.x];
    }
#endif

// == Determine the spot center from the convolution result
    // There's not much to be parallelized here, so let's do it single threaded.
    if (threadIdx.x == 0)
    {
        // The index of max. convolution value, operates on the 1D array of convResults
        int maxPosIdx = 0; 
        // The maximum of the convolution value
        float maxVal = convResults[maxPosIdx];
        float curVal = maxVal;
        float signalSum = 0;
        for (int i = 0; i < GL->mNumCorrelPosPerAp; i++)
        {
            curVal = convResults[i];
            signalSum += curVal;
            if (maxVal < curVal)
            {
                maxVal = curVal;
                maxPosIdx = i;
            }
            // Copy the colvoluted values to the shm array of image data for 2D access
            // Yes, we overwrite the (GPU-)shared memory camera data, but we don't need it anymore.
            imData[convCoordsY[i]*GL->mWindowSize + convCoordsX[i]] = curVal;
        }
        // Compare the accumulated signal in the convolution area with the last value.
        // If the intensity drops suddenly, the measurement nay have lost track
        // of the spot and is considered invalid.
        // The searching position will be reset for the next frame in this case.
        float lastIntensity = d_intensityOut[blockIdx.x];
        measurementValid = lastIntensity * 0.4 < signalSum;
        d_intensityOut[blockIdx.x] = signalSum;
        // 2D index of the max. convolution value
        int maxIdxX = convCoordsX[maxPosIdx];
        int maxIdxY = convCoordsY[maxPosIdx];
        int centerIndex = GL->mWindowSize/2 - 1;

        // Check if the maximum position is within the center of the convolution region.
        // If it is not, the convolution window will be moved to get it back into the center.
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

#ifdef DEBUG_SHOW_MAX_PIXEL_COORDS
        // All the following actions happen in the debug image, but shifted by a
        // fixed number of pixels in x-direction to compare the result with the input.
        int lateralShitInPx = 5;
        // Generate dark area around the convolution area + 1 pixel in order to form a
        // dark rim for visual identification of the debug output 
        for (int i = 0; i < GL->mNumCorrelPosPerAp; i++)
        {
            int shiftedX = convCoordsX[i] + windowRootX + lateralShitInPx;
            int shiftedY = convCoordsY[i] + windowRootY;
            for (int outlineOffsetX = -1; outlineOffsetX < 2; outlineOffsetX++)
                for (int outlineOffsetY = -1; outlineOffsetY < 2; outlineOffsetY++)
                    d_debugImage[(shiftedY+outlineOffsetY)*imW + shiftedX + outlineOffsetX] = -maxVal/5;
        }
        // Generate a ramp pattern on the convolution pattern to map the pixel to
        // the convolution index
        for (int i = 0; i < GL->mNumCorrelPosPerAp; i++)
        {
            int shiftedX = convCoordsX[i] + windowRootX + lateralShitInPx;
            int shiftedY = convCoordsY[i] + windowRootY;
            d_debugImage[(shiftedY)*imW + shiftedX] = maxVal/(GL->mNumCorrelPosPerAp-1) * i;
        }
        // Make the pixel corresponding to the max pixel dark to distinguish it from the ramp pattern.
        int maxShiftedX = windowRootX + convCoordsX[maxPosIdx] + lateralShitInPx;
        int maxShiftedY = convCoordsY[maxPosIdx] + windowRootY;
        d_debugImage[(maxShiftedY)*imW + maxShiftedX] = -maxVal/5;
#endif

        // == Image evaluation done! Calculate the gradient: ==
        // globalSpotPosition = windowRoot + spotPosInWindow
        // spotShift = globalSpotPosition - ref
        // gradient = spotShift * shift2gradConst
        float shiftX = windowRootX + spotPosXInWindow - refX;
        d_gradOut[blockIdx.x] = shiftX * shift2gradConst;
        float shiftY = windowRootY + spotPosYInWindow - refY;
        d_gradOut[blockIdx.x + gridDim.x] = shiftY * shift2gradConst;
        // If the shift is too large, the measurement may have lost track of the spot
        // and is considered invalid.
        // The searching position will be reset for the next frame in this case.
        if (shiftX > outOfRangeDistance || shiftX < -outOfRangeDistance)
            measurementValid = false;
        if (shiftY > outOfRangeDistance || shiftY < -outOfRangeDistance)
            measurementValid = false;

        // Prepare for the next frame:
        // If the measurement is invalid, reset the center of the tracking rectangle.
        // Else, if the spot drifted out of the center of the tracking rectangle,
        // update the window root positions accordingly.
        if (!measurementValid)
            d_windowCentersX[blockIdx.x] = round(refX);
        else if (spotPosXInWindow < ((float)centerIndex))
            d_windowCentersX[blockIdx.x]--;
        else if (spotPosXInWindow > centerIndex+1.f)
            d_windowCentersX[blockIdx.x]++;

        if (!measurementValid)
            d_windowCentersY[blockIdx.x] = round(refY);
        else if (spotPosYInWindow < ((float)centerIndex))
            d_windowCentersY[blockIdx.x]--;
        else if (spotPosYInWindow > centerIndex+1.f)
            d_windowCentersY[blockIdx.x]++;

#ifdef DEBUG_ARRAY_OUTPUT
        if (threadIdx.x == 0)
        {
            // Adapt the contents of this array to your gusto.
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
#endif

    }
}
