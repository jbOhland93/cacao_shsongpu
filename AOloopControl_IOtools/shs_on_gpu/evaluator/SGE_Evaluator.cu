#include "SGE_Evaluator.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>

#include "SGE_GridLayout.hpp"
#include "SGE_Kernel.hpp"

#include "../util/atypeUtil.hpp"
#include "../ref_recorder/SGR_Recorder.hpp"

SGE_Evaluator::SGE_Evaluator(
        IMAGE* ref,                 // Stream holding the reference data
        IMAGE* cam,                 // Stream holding the current SHS frame
        IMAGE* dark,                // Stream holding the dark frame of the SHS
        const char* streamPrefix,   // Prefix for the ISIO streams
        int deviceID)               // ID of the GPU device
    : m_streamPrefix(streamPrefix), m_deviceID(deviceID)
{
    printf("TODO: remove SGE_Evaluator::emulateReferenceInput()\n");


    // Set up cuda environment
    cudaError err;
    err = cudaSetDevice(m_deviceID);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    printCE(err);

    // Load the reference
    mp_refManager = SGE_ReferenceManager::makeReferenceManager(
                                            ref,
                                            cam,
                                            dark,
                                            streamPrefix);
    // Adopt the image streams
    mp_IHcam = SGR_ImageHandler<uint16_t>::newHandlerAdoptImage(cam->name);
    mp_IHdark = SGR_ImageHandler<float>::newHandlerAdoptImage(dark->name);
    // Initialize spot search positions
    mp_refManager->initGPUSearchPositions(&mp_d_SearchPosX, &mp_d_SearchPosY);
    // Set up the grid layout for the cuda calls
    mp_GridLayout = SGE_GridLayout::makeGridLayout(
        m_deviceID, mp_refManager);
    // Prepare some fields for debugging
    initDebugFields();
}

errno_t SGE_Evaluator::evaluateDo()
{
    printf("\nStart evaluateDo\n\n");
    // Make sure the GPU copy of the darkframe is up to date!
    mp_IHdark->updateGPUCopy();

    // Prepare image buffer
// ====================================
    // Ultimately, the image source should already be a mapped pointer
    uint16_t* h_readBufBright;
    int bufSize = mp_IHcam->getBufferSize();
    cudaError_t err = cudaHostAlloc(&h_readBufBright, bufSize, cudaHostAllocMapped);
    printCE(err);
    err = cudaMemcpy((void**)h_readBufBright, mp_IHcam->getWriteBuffer(), bufSize, cudaMemcpyHostToHost);
    printCE(err);
// ====================================

    // Prepare the debug image buffer
    mp_IHdebug->cpy(mp_IHcam->getImage());
    mp_IHdebug->updateGPUCopy();

    // Do the shs evaluation.
    cudaDeviceSynchronize();
    err = cudaEventRecord( m_cuEvtStart, 0 );
    printCE(err);
    evaluateSpots<<<mp_GridLayout->mNumSubapertures,
                    mp_GridLayout->mBlockSize,
                    mp_GridLayout->mShmSize>>>(
            h_readBufBright,                    //uint16_t* h_imageData,
            mp_IHdark->getGPUCopy(),            //float* d_darkData,
            mp_IHcam->mWidth,                   //int imW,
            mp_GridLayout->getDeviceCopy(),     //SGE_GridLayout* d_GridLayout,
            mp_d_SearchPosX,                    //int windowRootX,
            mp_d_SearchPosY,                    //int windowRootY,
            mp_refManager->getKernelBufferGPU(),//float* d_kernel,
            mp_GridLayout->mp_d_CorrelationOffsetsX,                   //int* d_convCoordsX,
            mp_GridLayout->mp_d_CorrelationOffsetsY,                   //int* d_convCoordsY,
            mp_IHdebug->getGPUCopy(),        //float* device copy of debug image
            mp_d_debug                          //float* debug
            );
    err = cudaDeviceSynchronize();
    printCE(err);
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

    cudaMemcpy(mp_h_debug, mp_d_debug, m_debugBufSize*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Debug 0: %.6f\n", mp_h_debug[0]);
    printf("Debug 1: %.6f\n", mp_h_debug[1]);
    printf("Debug 2: %.6f\n", mp_h_debug[2]);
    printf("Debug 3: %.6f\n", mp_h_debug[3]);
    printf("Debug 4: %.6f\n", mp_h_debug[4]);
    printf("Debug 5: %.6f\n", mp_h_debug[5]);
    printf("Debug 6: %.6f\n", mp_h_debug[6]);
    printf("Debug 7: %.6f\n", mp_h_debug[7]);
    printf("Debug 8: %.6f\n", mp_h_debug[8]);
    printf("Debug 9: %.6f\n", mp_h_debug[9]);
    mp_IHdebug->updateFromGPU();


// Clean up once the image is permanently in device memory
// =================
    cudaFreeHost(h_readBufBright);
// =================

    printf("\nEnd evaluationDo\n\n");
    return RETURN_SUCCESS;
}

SGE_Evaluator::~SGE_Evaluator()
{
    // Free arrays
    cudaFree(mp_d_SearchPosX);
    cudaFree(mp_d_SearchPosY);

    // Clean up debugging fields
    cudaEventDestroy(m_cuEvtStart);
    cudaEventDestroy(m_cuEvtStop);
    cudaFreeHost(mp_h_debug);
    cudaFree(mp_d_debug);
}

void SGE_Evaluator::initDebugFields()
{
    printf("Initializing debug fields ...\n");
    cudaError_t err = cudaEventCreate(&m_cuEvtStart);
    err = cudaEventCreate(&m_cuEvtStop);
    printCE(err);

    // Generate the debug image
    std::string debugImgName = m_streamPrefix;
    debugImgName.append("DEBUG");
    mp_IHdebug = SGR_ImageHandler<float>::newImageHandler(
        debugImgName.c_str(), mp_IHdark->mWidth, mp_IHdark->mHeight);
    mp_IHdebug->setPersistent(true);

    // Set a size for the debug buffer.
    // Currently: Two fields per convolution position.
    m_debugBufSize = mp_GridLayout->mNumCorrelPosPerAp*2;
    err = cudaHostAlloc(&mp_h_debug, m_debugBufSize*sizeof(float), cudaHostAllocMapped);
    printCE(err);
    err = cudaMalloc(&mp_d_debug, m_debugBufSize*sizeof(float));
    printCE(err);
}
