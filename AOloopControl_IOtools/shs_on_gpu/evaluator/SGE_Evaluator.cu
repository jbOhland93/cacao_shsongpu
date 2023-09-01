#include "SGE_Evaluator.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>

#include "SGE_GridLayout.hpp"
#include "SGE_Kernel.hpp"

#include "../util/atypeUtil.hpp"
#include "../ref_recorder/SGR_Recorder.hpp"

// If this is true, a warning will be printed when the camera image
// does not reside in mapped memory (highly recommended for speed).
#define WARN_IF_NO_MAPPED_CAM_IMG true

// === DEBUGGING FLAGS ===
// Visual debugging - generate a debug image to visually inspect the evaluation steps.
// Note: In order for useful output, enable visual debugging flags in SGE_KERNEL.hpp as well.
//#define ENABLE_DEBUG_IMAGE

// Commandline debugging - catch values from the cuda kernel and print them.
// Note: In order for useful output, enable visual debugging flags in SGE_KERNEL.hpp as well.
//#define ENABLE_DEBUG_ARRAY

// Record and print the time for each evaluation.
// Note: This includes all debugging and copying processes.
//#define ENABLE_EVALUATION_TIME_PRINTING

SGE_Evaluator::SGE_Evaluator(
        IMAGE* ref,                 // Stream holding the reference data
        IMAGE* cam,                 // Stream holding the current SHS frame
        IMAGE* dark,                // Stream holding the dark frame of the SHS
        const char* streamPrefix,   // Prefix for the ISIO streams
        int deviceID)               // ID of the GPU device
    : m_streamPrefix(streamPrefix), m_deviceID(deviceID)
{
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

SGE_Evaluator::~SGE_Evaluator()
{
    // Free arrays
    cudaFree(mp_d_SearchPosX);
    cudaFree(mp_d_SearchPosY);
    if (mp_h_camArrayMappedCpy != nullptr);
        cudaFreeHost(mp_h_camArrayMappedCpy);

    // Clean up debugging fields
#ifdef ENABLE_EVALUATION_TIME_PRINTING
    cudaEventDestroy(m_cuEvtStart);
    cudaEventDestroy(m_cuEvtStop);
#endif
#ifdef ENABLE_DEBUG_ARRAY
    cudaFreeHost(mp_h_debug);
    cudaFree(mp_d_debug);
#endif
}

errno_t SGE_Evaluator::evaluateDo()
{
    // Measure the evaluation time
    // (if ENABLE_EVALUATION_TIME_PRINTING is defined)
    startRecordingTime();

    // Make sure the GPU copy of the darkframe is up to date!
    mp_IHdark->updateGPUCopy();
    // Wait until all copying is done!
    cudaDeviceSynchronize();

    // Kernel call - evaluation happens here!
    evaluateSpots<<<mp_GridLayout->mNumSubapertures,
                    mp_GridLayout->mBlockSize,
                    mp_GridLayout->mShmSize>>>(
            getCamInputArrPtr(),                    //uint16_t* h_imageData,
            mp_IHdark->getGPUCopy(),            //float* d_darkData,
            mp_IHcam->mWidth,                   //int imW,
            mp_GridLayout->getDeviceCopy(),     //SGE_GridLayout* d_GridLayout,
            mp_d_SearchPosX,                    //int windowRootX,
            mp_d_SearchPosY,                    //int windowRootY,
            mp_refManager->getKernelBufferGPU(),//float* d_kernel,
            mp_GridLayout->mp_d_CorrelationOffsetsX,                   //int* d_convCoordsX,
            mp_GridLayout->mp_d_CorrelationOffsetsY,                   //int* d_convCoordsY,
            prepareDebugImageDevicePtr(),        //float* device copy of debug image
            mp_d_debug                          //float* debug
            );
    cudaDeviceSynchronize();
    // Print error of kernel launch
    printCE(cudaGetLastError());

    // If any debug flags are defined, collect data from
    // devide memory and make them accessible / print them
    provideDebugOutputAfterEval();

    return RETURN_SUCCESS;
}

uint16_t* SGE_Evaluator::getCamInputArrPtr()
{
    printf("TODO SGE_Evaluator::getCamInputArrPtr: Provide actual check for mapped memory.\n");
    bool imageInMappedMem = false;
    if (imageInMappedMem)
        return mp_IHcam->getWriteBuffer();
    else
    {
        if (WARN_IF_NO_MAPPED_CAM_IMG)
        {
            printf("\n### WARNING: ###\n");
            printf("The camera image is not located in mapped memory! ");
            printf("Therefore, the data will be copied to a freshly ");
            printf("allocated mapped memory array. This is expensive ");
            printf("and will greatly slow down the process. Make sure ");
            printf("to stream the frames from the camera directly into ");
            printf("mapped memory.\n\n");
        }

        cudaError_t err;
        int bufSize = mp_IHcam->getBufferSize();
        if (mp_h_camArrayMappedCpy == nullptr)
        {
            err = cudaHostAlloc(&mp_h_camArrayMappedCpy, bufSize, cudaHostAllocMapped);
            printCE(err);
        }
        err = cudaMemcpy((void**)mp_h_camArrayMappedCpy, mp_IHcam->getWriteBuffer(), bufSize, cudaMemcpyHostToHost);
        printCE(err);
        return mp_h_camArrayMappedCpy;
    }
}

void SGE_Evaluator::initDebugFields()
{
#ifdef ENABLE_EVALUATION_TIME_PRINTING
    // Create cuda events for timing
    printf("Initializing cuda events for evaluation time printing ...\n");
    cudaError_t err = cudaEventCreate(&m_cuEvtStart);
    err = cudaEventCreate(&m_cuEvtStop);
    printCE(err);
#endif

#ifdef ENABLE_DEBUG_IMAGE
    // Generate the debug image
    printf("Initializing debugging image for visual inspection of evaluation steps ...\n");
    std::string debugImgName = m_streamPrefix;
    debugImgName.append("DEBUG");
    mp_IHdebug = SGR_ImageHandler<float>::newImageHandler(
        debugImgName.c_str(), mp_IHdark->mWidth, mp_IHdark->mHeight);
    mp_IHdebug->setPersistent(true);
#endif

#ifdef ENABLE_DEBUG_ARRAY
    printf("Initializing debugging array for printing of kernel valus ...\n");
    // Set a size for the debug buffer.
    // Currently: Two fields per convolution position.
    m_debugBufSize = mp_GridLayout->mNumCorrelPosPerAp*2;
    cudaError_t err = cudaHostAlloc(&mp_h_debug, m_debugBufSize*sizeof(float), cudaHostAllocMapped);
    printCE(err);
    err = cudaMalloc(&mp_d_debug, m_debugBufSize*sizeof(float));
    printCE(err);
#endif
}

void SGE_Evaluator::startRecordingTime()
{
#ifdef ENABLE_EVALUATION_TIME_PRINTING
    cudaDeviceSynchronize();
    printCE(cudaEventRecord( m_cuEvtStart, 0 ));
#endif
}

float* SGE_Evaluator::prepareDebugImageDevicePtr()
{
    float* d_debugImageArr = nullptr;
#ifdef ENABLE_DEBUG_IMAGE
    // Prepare the debug image buffer
    mp_IHdebug->cpy(mp_IHcam->getImage());
    d_debugImageArr = mp_IHdebug->updateGPUCopy();
#endif
    return d_debugImageArr;
}

void SGE_Evaluator::provideDebugOutputAfterEval()
{
#ifdef ENABLE_EVALUATION_TIME_PRINTING
    // Get timing
    cudaError_t err = cudaEventRecord( m_cuEvtStop, 0 );
    printCE(err);
    err = cudaEventSynchronize( m_cuEvtStop );
    printCE(err);
    float time;
    err = cudaEventElapsedTime( &time, m_cuEvtStart, m_cuEvtStop );
    printCE(err);
    printf("Time for kernel call: %.3f µs\n", time*1000);
#endif

#ifdef ENABLE_DEBUG_ARRAY
    // Copy the debug array from the device and print values.
    // Adapt / Add descriptions as needed.
    cudaError_t err cudaMemcpy(mp_h_debug, mp_d_debug, m_debugBufSize*sizeof(float), cudaMemcpyDeviceToHost);
    printCE(err);
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
#endif

#ifdef ENABLE_DEBUG_IMAGE
    mp_IHdebug->updateFromGPU();
#endif
}
