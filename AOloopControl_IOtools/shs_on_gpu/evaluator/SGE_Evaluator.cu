#include "SGE_Evaluator.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>

#include "SGE_GridLayout.hpp"
#include "SGE_Kernel.hpp"
#include "SGE_ReferenceManager.hpp"
#include "../util/atypeUtil.hpp"
#include "../ref_recorder/SGR_Recorder.hpp"

SGE_Evaluator::SGE_Evaluator(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Stream holding the current SHS frame
        IMAGE* dark,        // Stream holding the dark frame of the SHS
        int deviceID)       // ID of the GPU device
    : m_deviceID(deviceID)
{
    printf("\n\n\n\n\n\nSTARTING CHECK!!!!!! - deviceid = %d\n\n", deviceID);
    SGE_ReferenceManager manager(ref, cam, dark, deviceID);
    
    
    /*spImageHandler<
    ref->kw
    Also, make a facfun for an image handler, taking only the pointer to the image as input. Something like "adoptImage".
    This makes browsing the keywords a lot easier ...

    TODO: read keywords from ref, resolve images, check cam stream against trigger stream!
    Set mp_im(in), mp_imDark(dark), if everything matches.
*/

    cudaError err;
    err = cudaSetDevice(m_deviceID);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    printCE(err);

    //emulateReferenceInput();

    //mp_GridLayout = std::make_shared<SGE_GridLayout>(m_deviceID, m_numSpots, m_kernelSize);
    //writeApertureConvolutionCoordToGPU();
    //copyDarkToGPU();
    //initDebugFields();
}

errno_t SGE_Evaluator::evaluateDo()
{
    printf("\nStart evaluateDo\n\n");
    // Prepare image buffer
// ====================================
    // Ultimately, the image source should already be a mapped pointer
    uint16_t* readBufBrightNonMapped;
    ImageStreamIO_readLastWroteBuffer(mp_im, (void**)&readBufBrightNonMapped);
    uint16_t* h_readBufBright;
    int bufSize = mp_im->md->size[0]*mp_im->md->size[1]*sizeof(uint16_t);
    cudaError_t err = cudaHostAlloc(&h_readBufBright, bufSize, cudaHostAllocMapped);
    printCE(err);
    err = cudaMemcpy((void**)h_readBufBright, readBufBrightNonMapped, bufSize, cudaMemcpyHostToHost);
    printCE(err);
// ====================================

    // Prepare dark buffer - the image already is on the GPU.
    float* d_readBufDark;
    ImageStreamIO_readLastWroteBuffer(&m_imDarkGPU, (void**)&d_readBufDark);

    // Prepare the debug image buffer
    float* d_debugIm = copyDebugImgToGPU();


    //cudaDeviceSynchronize();
    err = cudaEventRecord( m_cuEvtStart, 0 );
    printCE(err);
    evaluateSpots<<<mp_GridLayout->mNumSubapertures,
                    mp_GridLayout->mBlockSize,
                    mp_GridLayout->mShmSize>>>(
            h_readBufBright,                    //uint16_t* h_imageData,
            d_readBufDark,                      //float* d_darkData,
            mp_im->md->size[0],                 //int imW,
            mp_GridLayout->getDeviceCopy(),     //SGE_GridLayout* d_GridLayout,
            m_dp_ROIx,                          //int windowRootX,
            m_dp_ROIy,                          //int windowRootY,
            m_dp_kernel,                        //float* d_kernel,
            m_dp_convCoordsX,                   //int* d_convCoordsX,
            m_dp_convCoordsY,                   //int* d_convCoordsY,
            d_debugIm,
            m_dp_debug                          //float* debug
            );
    cudaDeviceSynchronize();
    err = cudaEventRecord( m_cuEvtStop, 0 );
    printCE(err);

    // Get timing
    err = cudaEventSynchronize( m_cuEvtStop );
    printCE(err);
    float time;
    err = cudaEventElapsedTime( &time, m_cuEvtStart, m_cuEvtStop );
    printCE(err);
    printf("Time for kernel call: %.3f µs\n", time*1000);

    // Print error of kernel launch
    printf("Assessing errors of kernel launch ... ");
    //err = cudaDeviceSynchronize();
    //printCE(err);
    err = cudaGetLastError();
    printCE(err);
    printf("Assessing done!\n");

    cudaMemcpy(m_hp_debug, m_dp_debug, m_debugBufSize*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Debug 0: %.6f\n", m_hp_debug[0]);
    printf("Debug 1: %.6f\n", m_hp_debug[1]);
    printf("Debug 2: %.6f\n", m_hp_debug[2]);
    printf("Debug 3: %.6f\n", m_hp_debug[3]);
    printf("Debug 4: %.6f\n", m_hp_debug[4]);
    printf("Debug 5: %.6f\n", m_hp_debug[5]);
    printf("Debug 6: %.6f\n", m_hp_debug[6]);
    printf("Debug 7: %.6f\n", m_hp_debug[7]);
    printf("Debug 8: %.6f\n", m_hp_debug[8]);
    printf("Debug 9: %.6f\n", m_hp_debug[9]);
    copyDebugImgFrmGPU();


// Clean up
// =================
    cudaFreeHost(h_readBufBright);
// =================

    printf("\nEnd evaluationDo\n\n");
    return RETURN_SUCCESS;
}

SGE_Evaluator::~SGE_Evaluator()
{
    // Delete the darkframe from the GPU
    ImageStreamIO_destroyIm(&m_imDarkGPU);
    // Free arrays
    cudaFreeHost(m_hp_ROIx);
    cudaFreeHost(m_hp_ROIy);
    cudaFree(m_dp_ROIx);
    cudaFree(m_dp_ROIy);
    cudaFree(m_dp_kernel);
    cudaFree(m_dp_convCoordsX);
    cudaFree(m_dp_convCoordsY);

    // Clean up debugging fields
    cudaEventDestroy(m_cuEvtStart);
    cudaEventDestroy(m_cuEvtStop);
    ImageStreamIO_destroyIm(&m_imDebug);
    ImageStreamIO_destroyIm(&m_imDebug_GPU);
    cudaFreeHost(m_hp_debug);
    cudaFree(m_dp_debug);
}

void SGE_Evaluator::emulateReferenceInput()
{
    // Create some tile coordinates, which ultimately live on the GPU forever
    uint16_t tileGridSize = 20;
    int offsetX = 2;
    int offsetY = 0;


    m_numSpots = tileGridSize * tileGridSize;
    int bufSize = m_numSpots*sizeof(uint16_t);
    cudaHostAlloc((void**)&m_hp_ROIx, bufSize, cudaHostAllocMapped);
    cudaHostAlloc((void**)&m_hp_ROIy, bufSize, cudaHostAllocMapped);
    for (uint16_t x = 0; x < tileGridSize; x++)
        for (uint16_t y = 0; y < tileGridSize; y++)
        {
            int xPre = (x+offsetX) * 18 + 20; // randomize;
            m_hp_ROIx[x+y*tileGridSize] = xPre < 0 ? 0 : xPre;
            int yPre = (y+offsetY) * 18 + 29; // randomize;
            m_hp_ROIy[x+y*tileGridSize] = yPre < 0 ? 0 : yPre;
        }
    cudaMalloc((void**)&m_dp_ROIx, bufSize);
    cudaMalloc((void**)&m_dp_ROIy, bufSize);
    cudaMemcpy(m_dp_ROIx, m_hp_ROIx, bufSize, cudaMemcpyHostToDevice);
    cudaMemcpy(m_dp_ROIy, m_hp_ROIy, bufSize, cudaMemcpyHostToDevice);

    // Create a block kernel, which ultimately lives on the GPU
    m_kernelSize = 3;
    float kernelSigma = 0.7;
    int kernelCenter = m_kernelSize / 2;
    int numKernelPixels = m_kernelSize*m_kernelSize;
    float* h_kernel;
    cudaHostAlloc((void**)&h_kernel, numKernelPixels*sizeof(float), cudaHostAllocMapped);
    for (uint16_t x = 0; x < m_kernelSize; x++)
        for (uint16_t y = 0; y < m_kernelSize; y++)
        {
            /*if (x==0 && y==0)
                h_kernel[x+y*m_kernelSize] = 1.f/numKernelPixels;
            else
                h_kernel[x+y*m_kernelSize] = 1.f/numKernelPixels;*/
            h_kernel[x+y*m_kernelSize] = 0.5*exp(
                -(pow(x-kernelCenter,2) + pow(y-kernelCenter,2))
                / (2*pow(kernelSigma, 2))
                );
        }
    
    cudaMalloc((void**)&m_dp_kernel, numKernelPixels*sizeof(float));
    cudaMemcpy(m_dp_kernel, h_kernel, numKernelPixels*sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h_kernel);
}

void SGE_Evaluator::writeApertureConvolutionCoordToGPU()
{
    int convolutionsPerAperture = SGE_GridLayout::getNumConvolutionsPerAp();
    int convCoordsX[convolutionsPerAperture];
    int convCoordsY[convolutionsPerAperture];
    mp_GridLayout->gnrtCorrelOffsetsFrmRoots(convCoordsX, convCoordsY);
    cudaMalloc(&m_dp_convCoordsX, convolutionsPerAperture*sizeof(int));
    cudaMalloc(&m_dp_convCoordsY, convolutionsPerAperture*sizeof(int));
    cudaMemcpy(m_dp_convCoordsX, convCoordsX, convolutionsPerAperture*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dp_convCoordsY, convCoordsY, convolutionsPerAperture*sizeof(int), cudaMemcpyHostToDevice);
}

void SGE_Evaluator::copyDarkToGPU()
{
    printf("Copying darkframe to GPU ...\n");
    std::string imName(mp_imDark->name);
    imName.append("_GPU");

    ImageStreamIO_createIm_gpu(
        &m_imDarkGPU,
        imName.c_str(),
        mp_imDark->md->naxis,
        mp_imDark->md->size,
        mp_imDark->md->datatype,
        m_deviceID,         // -1: CPU RAM, 0+ : GPU
        1,                  // shared?
        0,                  // # of semaphores
        0,                  // # of keywords
        mp_imDark->md->imagetype,
        0 // circular buffer size (if shared), 0 if not used
    );

    float* readBuf;
    ImageStreamIO_readLastWroteBuffer(mp_imDark, (void**)&readBuf);
    float* writeBuf;
    ImageStreamIO_writeBuffer(&m_imDarkGPU, (void**)&writeBuf);
    
    cudaError_t err;
    err = cudaMemcpy(writeBuf, readBuf, mp_imDark->md->imdatamemsize,
        cudaMemcpyHostToDevice);
    printCE(err);
    ImageStreamIO_UpdateIm(&m_imDarkGPU);
}

void SGE_Evaluator::initDebugFields()
{
    printf("Initializing debug fields ...\n");
    cudaError_t err = cudaEventCreate(&m_cuEvtStart);
    err = cudaEventCreate(&m_cuEvtStop);
    printCE(err);

    ImageStreamIO_createIm_gpu(
        &m_imDebug,
        "shsOnGPU-debug",
        m_imDarkGPU.md->naxis,
        m_imDarkGPU.md->size,
        m_imDarkGPU.md->datatype,
        -1,                 // -1: CPU RAM, 0+ : GPU
        1,                  // shared?
        0,                  // # of semaphores
        0,                  // # of keywords
        m_imDarkGPU.md->imagetype,
        0 // circular buffer size (if shared), 0 if not used
    );
    ImageStreamIO_createIm_gpu(
        &m_imDebug_GPU,
        "shsOnGPU-debug_GPU",
        m_imDarkGPU.md->naxis,
        m_imDarkGPU.md->size,
        m_imDarkGPU.md->datatype,
        m_deviceID,         // -1: CPU RAM, 0+ : GPU
        1,                  // shared?
        0,                  // # of semaphores
        0,                  // # of keywords
        m_imDarkGPU.md->imagetype,
        0 // circular buffer size (if shared), 0 if not used
    );

    // Set a size for the debug buffer.
    // Currently: Two fields per convolution position.
    m_debugBufSize = mp_GridLayout->mNumCorrelPosPerAp*2;
    err = cudaHostAlloc(&m_hp_debug, m_debugBufSize*sizeof(float), cudaHostAllocMapped);
    printCE(err);
    err = cudaMalloc(&m_dp_debug, m_debugBufSize*sizeof(float));
    printCE(err);
}

float* SGE_Evaluator::copyDebugImgToGPU()
{
    float* h_darksub;
    cudaHostAlloc(&h_darksub, mp_imDark->md->imdatamemsize, cudaHostAllocMapped);
    int numPx = mp_im->md->size[0]*mp_im->md->size[1];

    uint16_t* im;
    ImageStreamIO_readLastWroteBuffer(mp_im, (void**)&im);
    float* dark;
    ImageStreamIO_readLastWroteBuffer(mp_imDark, (void**)&dark);
    for (int i = 0; i < numPx; i++)
        h_darksub[i] = im[i] - dark[i];
    
    float* d_darksub;
    ImageStreamIO_writeBuffer(&m_imDebug_GPU, (void**)&d_darksub);
    cudaMemcpy(d_darksub, h_darksub, mp_imDark->md->imdatamemsize, cudaMemcpyHostToDevice);
    ImageStreamIO_UpdateIm(&m_imDebug_GPU);

    cudaFreeHost(h_darksub);

    return d_darksub;
}
float* SGE_Evaluator::copyDebugImgFrmGPU()
{
    float* d_debug;
    ImageStreamIO_writeBuffer(&m_imDebug_GPU, (void**)&d_debug);
    float* h_debug;
    ImageStreamIO_writeBuffer(&m_imDebug, (void**)&h_debug);
    
    cudaMemcpy(h_debug, d_debug, m_imDebug.md->imdatamemsize, cudaMemcpyDeviceToHost);
    ImageStreamIO_UpdateIm(&m_imDebug);

    return h_debug;
}
