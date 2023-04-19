#include "SGE_Evaluator.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>

#include "SGE_GridLayout.hpp"
#include "SGE_Kernel.hpp"

#include <chrono>

using namespace std::chrono;

SGE_Evaluator::SGE_Evaluator(
        IMAGE* in,          // Raw camera stream
        IMAGE* dark,        // Stream holding a dark for subtraction
        int deviceID)       // ID of the GPU
    : mp_im(in), m_deviceID(deviceID)
{
    cudaError err;
    err = cudaSetDevice(m_deviceID);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printCE(err);

    copyDarkToGPU(dark);

    // Create some tile coordinates, which ultimately live on the GPU forever
    uint16_t tileGridSize = 20;
    int bufSize = tileGridSize*tileGridSize*sizeof(uint16_t);
    cudaHostAlloc((void**)&m_hp_ROIx, bufSize, cudaHostAllocMapped);
    cudaHostAlloc((void**)&m_hp_ROIy, bufSize, cudaHostAllocMapped);
    for (uint16_t x = 0; x < tileGridSize; x++)
        for (uint16_t y = 0; y < tileGridSize; y++)
        {
            int xPre = x * 16 + rand()%3-1 + 20; // randomize;
            m_hp_ROIx[x+y*tileGridSize] = xPre < 0 ? 0 : xPre;
            int yPre = y * 16 + rand()%3-1 + 20; // randomize;
            m_hp_ROIy[x+y*tileGridSize] = yPre < 0 ? 0 : yPre;
        }
    cudaMalloc((void**)&m_dp_ROIx, bufSize);
    cudaMalloc((void**)&m_dp_ROIy, bufSize);
    cudaMemcpy(m_dp_ROIx, m_hp_ROIx, bufSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_dp_ROIy, m_hp_ROIy, bufSize, cudaMemcpyHostToDevice);

    // Create a block kernel, which ultimately lives on the GPU
    int kernelSize = 3;
    float* h_kernel;
    cudaHostAlloc((void**)&h_kernel, kernelSize*kernelSize*sizeof(float), cudaHostAllocMapped);
    for (uint16_t x = 0; x < kernelSize; x++)
        for (uint16_t y = 0; y < kernelSize; y++)
        {
            if (x==1 && y==1)
                h_kernel[x+y*kernelSize] = 0.111;
            else
                h_kernel[x+y*kernelSize] = 0.111;
        }
    cudaMalloc((void**)&m_dp_kernel, kernelSize*kernelSize*sizeof(float));
    cudaMemcpy(m_dp_kernel, h_kernel, kernelSize*kernelSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h_kernel);

    mpGridLayout = std::make_shared<SGE_GridLayout>(m_deviceID, tileGridSize*tileGridSize, kernelSize, 4);
}

errno_t SGE_Evaluator::evaluateDo()
{
    // ==== Prepare timing utilities
    cudaEvent_t start, stop;
    float time;
    cudaError_t err = cudaEventCreate(&start);
    printCE(err);
    err = cudaEventCreate(&stop);
    printCE(err);
    // ====



    printf("\nStart evaluateDo\n\n");
    // Prepare image buffer
    // Ultimately, the image source should already be a mapped pointer
    uint16_t* readBufBrightNonMapped;
    ImageStreamIO_readLastWroteBuffer(mp_im, (void**)&readBufBrightNonMapped);
    uint16_t* h_readBufBright;
    int bufSize = mp_im->md->size[0]*mp_im->md->size[1]*sizeof(uint16_t);
    err = cudaHostAlloc(&h_readBufBright, bufSize, cudaHostAllocMapped);
    printCE(err);
    err = cudaMemcpy((void**)h_readBufBright, readBufBrightNonMapped, bufSize, cudaMemcpyHostToHost);
    printCE(err);

    // Prepare dark buffer - the image already is on the GPU.
    float* d_readBufDark;
    ImageStreamIO_readLastWroteBuffer(&m_imDarkGPU, (void**)&d_readBufDark);

    // Prepare result buffer
    float* h_intensity;
    float* d_intensity;
    err = cudaHostAlloc(&h_intensity, mpGridLayout->mNumSubapertures*sizeof(float), cudaHostAllocMapped);
    printCE(err);
    err = cudaMalloc(&d_intensity, mpGridLayout->mNumSubapertures*sizeof(float));
    printCE(err);
    
    float* h_convImage;
    IMAGE convImage;
    ImageStreamIO_openIm(&convImage, "ximeaCam_AVG2");
    ImageStreamIO_writeBuffer(&convImage, (void**)&h_convImage);

    float* h_darkPx;
    cudaHostAlloc(&h_darkPx, m_imDarkGPU.md->imdatamemsize, cudaHostAllocMapped);
    cudaMemcpy(h_darkPx, d_readBufDark, m_imDarkGPU.md->imdatamemsize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < mp_im->md->size[0]*mp_im->md->size[1]; i++)
    {
        h_convImage[i] = (float) h_readBufBright[i] - h_darkPx[i];
    }
    cudaFreeHost(h_darkPx);
    ImageStreamIO_UpdateIm(&convImage);
    float* d_convImage;
    cudaMalloc(&d_convImage, convImage.md->imdatamemsize);
    cudaMemcpy(d_convImage, h_convImage, convImage.md->imdatamemsize, cudaMemcpyHostToDevice);

    err = cudaEventRecord( start, 0 );
    printCE(err);

    // Do processing
/*  evaluateSpot<<<
        mpGridLayout->mNumSubapertures,
        mpGridLayout->mBlockSize,
        mpGridLayout->mShmSize>>>(
            h_readBufBright,
            d_readBufDark,
            m_dp_kernel,
            mp_im->md->size[0],
            m_dp_ROIx,
            m_dp_ROIy,
            mpGridLayout->getDeviceCopy(),
            d_intensity,
            d_convImage);
*/
// ============================================
    cudaStream_t streams[mpGridLayout->mNumSubapertures];
    for (int i = 0; i < mpGridLayout->mNumSubapertures; ++i)
        cudaStreamCreate(&streams[i]);
    
    int blockSize = mpGridLayout->mKernelSize;
    blockSize *= blockSize;
    blockSize *= mpGridLayout->mNumCorrelPosPerAp;
    blockSize /= 32;
    blockSize++;
    blockSize *= 32;

    int numPxInWindow = mpGridLayout->mNumWindowPx;
    int numPxInKernel = mpGridLayout->mNumKernelPx;
    int numCorrelPos = mpGridLayout->mNumCorrelPosPerAp;
    int numCorrelOps = numCorrelPos * numPxInKernel;
    int shmSize = numPxInWindow + numPxInKernel + 2*numCorrelPos + numCorrelOps;
    shmSize *= sizeof(float);

    int convCoordsX[numCorrelPos];
    int convCoordsY[numCorrelPos];
    mpGridLayout->gnrtCorrelOffsetsFrmRootsHost(convCoordsX, convCoordsY);
    int* d_convCoordsX;
    int* d_convCoordsY;
    cudaMalloc(&d_convCoordsX, numCorrelPos*sizeof(int));
    cudaMalloc(&d_convCoordsY, numCorrelPos*sizeof(int));
    cudaMemcpy(d_convCoordsX, convCoordsX, numCorrelPos*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convCoordsY, convCoordsY, numCorrelPos*sizeof(int), cudaMemcpyHostToDevice);

    float* d_debug;
    int debugBufSize = 128;
    cudaMalloc(&d_debug, debugBufSize*sizeof(float));

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    /*One launch per block
    for (int i = 0; i < mpGridLayout->mNumSubapertures; ++i)
        evaluateSingleSpot<<<1, blockSize, shmSize, streams[i]>>>(
            h_readBufBright,                //uint16_t* h_imageData,
            d_readBufDark,                  //float* d_darkData,
            d_convImage,                    //float* d_outputImage,
            mp_im->md->size[0],             //int imW,
            m_hp_ROIx[i],                   //int windowRootX,
            m_hp_ROIy[i],                   //int windowRootY,
            mpGridLayout->mWindowSize,      //int windowSize,
            m_dp_kernel,                    //float* d_kernel,
            mpGridLayout->mKernelSize,      //int kernelSize,
            d_convCoordsX,                  //int* d_convCoordsX,
            d_convCoordsY,                  //int* d_convCoordsY,
            mpGridLayout->mNumCorrelPosPerAp//int numConvCoords)
            );
    */
    evaluateSpots<<<mpGridLayout->mNumSubapertures, blockSize, shmSize>>>(
            h_readBufBright,                    //uint16_t* h_imageData,
            d_readBufDark,                      //float* d_darkData,
            d_convImage,                        //float* d_outputImage,
            mp_im->md->size[0],                 //int imW,
            m_dp_ROIx,                          //int windowRootX,
            m_dp_ROIy,                          //int windowRootY,
            mpGridLayout->mWindowSize,          //int windowSize,
            m_dp_kernel,                        //float* d_kernel,
            mpGridLayout->mKernelSize,          //int kernelSize,
            d_convCoordsX,                      //int* d_convCoordsX,
            d_convCoordsY,                      //int* d_convCoordsY,
            mpGridLayout->mNumCorrelPosPerAp,   //int numConvCoords,
            mpGridLayout->mCorrelMargin,        //int correlMargin
            d_debug                             //float* debug
            );
    
    cudaDeviceSynchronize();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    printf("Time for kernel call: %.3f µs\n", time_span.count()*1e6);
    
    float* h_debug;
    cudaHostAlloc(&h_debug, debugBufSize*sizeof(float), cudaHostAllocMapped);
    cudaMemcpy(h_debug, d_debug, debugBufSize*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Debug 0: %.3f\n", h_debug[0]);
    printf("Debug 1: %.3f\n", h_debug[1]);
    printf("Debug 2: %.3f\n", h_debug[2]);
    printf("Debug 3: %.3f\n", h_debug[3]);
    printf("Debug 4: %.3f\n", h_debug[4]);
    printf("Debug 5: %.3f\n", h_debug[5]);
    printf("Debug 6: %.3f\n", h_debug[6]);
    printf("Debug 7: %.3f\n", h_debug[7]);
    printf("Debug 8: %.3f\n", h_debug[8]);
    printf("Debug 9: %.3f\n", h_debug[9]);
    
    
// ============================================

    err = cudaEventRecord( stop, 0 );
    printCE(err);

    // Get timing
    err = cudaEventSynchronize( stop );
    printCE(err);
    err = cudaEventElapsedTime( &time, start, stop );
    printCE(err);
    //printf("Time for kernel call: %.3f µs\n", time*1000);

    // Print error of kernel launch
    printf("Assessing errors of kernel launch ... ");
    //err = cudaDeviceSynchronize();
    //printCE(err);
    err = cudaGetLastError();
    printCE(err);
    printf("Assessing done!\n");

    // Copy result
    /*err = cudaMemcpy(h_intensity, d_intensity, mpGridLayout->mNumSubapertures*sizeof(float), cudaMemcpyDeviceToHost);
    printCE(err);
    printf("CorrelOPs pT = %.3f\n", h_intensity[5]);
    printf("CorrelPos pA = %.3f\n\n", h_intensity[6]);
    printf("Thread Index = %.3f\n", h_intensity[0]);
    printf("Apertr Index = %.3f\n", h_intensity[1]);
    printf("Coordn Index = %.3f\n", h_intensity[2]);
    printf("KernlY Index = %.3f\n", h_intensity[4]);
    printf("KernlX Index = %.3f\n\n", h_intensity[3]);*/

    // Copy convoluted image and post it to ISIO
    err = cudaMemcpy(h_convImage, d_convImage, convImage.md->imdatamemsize, cudaMemcpyDeviceToHost);
    printCE(err);
    ImageStreamIO_UpdateIm(&convImage);

    // Clean up
    cudaFreeHost(h_readBufBright);
    cudaFreeHost(h_intensity);
    cudaFree(d_intensity);
    cudaFree(d_convImage);
    cudaFree(d_convCoordsX);
    cudaFree(d_convCoordsY);
    cudaFree(d_debug);
    cudaFreeHost(h_debug);
    err = cudaEventDestroy( start );
    printCE(err);
    err = cudaEventDestroy( stop );
    printCE(err);

    return RETURN_SUCCESS;
}

SGE_Evaluator::~SGE_Evaluator()
{
    // Delete the darkframe from the GPU
    ImageStreamIO_destroyIm(&m_imDarkGPU);
    // Free array
    cudaFreeHost(m_hp_ROIx);
    cudaFreeHost(m_hp_ROIy);
    cudaFree(m_dp_ROIx);
    cudaFree(m_dp_ROIy);
    cudaFree(m_dp_kernel);
}

void SGE_Evaluator::copyDarkToGPU(IMAGE* dark)
{
    std::string imName(dark->name);
    imName.append("_GPU");

    ImageStreamIO_createIm_gpu(
        &m_imDarkGPU,
        imName.c_str(),
        dark->md->naxis,
        dark->md->size,
        dark->md->datatype,
        m_deviceID,         // -1: CPU RAM, 0+ : GPU
        1,                  // shared?
        0,                  // # of semaphores
        0,                  // # of keywords
        dark->md->imagetype,
        0 // circular buffer size (if shared), 0 if not used
    );

    float* readBuf;
    ImageStreamIO_readLastWroteBuffer(dark, (void**)&readBuf);
    float* writeBuf;
    ImageStreamIO_writeBuffer(&m_imDarkGPU, (void**)&writeBuf);
    
    cudaError_t err;
    err = cudaMemcpy(writeBuf, readBuf, dark->md->imdatamemsize,
        cudaMemcpyHostToDevice);
    printCE(err);
    ImageStreamIO_UpdateIm(&m_imDarkGPU);

    
}
