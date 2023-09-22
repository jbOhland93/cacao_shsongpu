#include "SGE_Evaluator.hpp"
#include "milkDebugTools.h"

#include <cuda.h>
#include <math.h>
#include <sstream>
#include <ctime>
#include <iomanip>

#include "SGE_GridLayout.hpp"
#include "SGE_CUDAkernel.hpp"

#include "../util/atypeUtil.hpp"
#include "../util/CudaUtil.hpp"

#include "../wf_reconstructor/modalwfreconstructorbuilder.hpp"


// ========== FLAGS ==========
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
#define ENABLE_EVALUATION_TIME_MEASUREMENT
// 0: Evaluation time will be printed to console
// 1: Evaluation time will be written to SGE_Eval_Times.dat in the current WD.
#define EVALTIME_PRINT_LOG 1
// ========== FLAGS END ==========


SGE_Evaluator::SGE_Evaluator(
        IMAGE* ref,                 // Stream holding the reference data
        IMAGE* cam,                 // Stream holding the current SHS frame
        IMAGE* dark,                // Stream holding the dark frame of the SHS
        const char* streamPrefix,   // Prefix for the ISIO streams
        int deviceID)               // ID of the GPU device
    : m_streamPrefix(streamPrefix)
{
    // Set up the evaluator
    setupCudaEnvironment(deviceID);
    setupReferenceManager(ref, cam, dark);
    adoptInputStreams(cam->name, dark->name);
    setupGridLayout(deviceID);
    createOutputImages();
    setupWFreconstruction();

    // Prepare some fields for debugging
    initDebugFields();
}

SGE_Evaluator::~SGE_Evaluator()
{
    // Clean up debugging fields
#ifdef ENABLE_EVALUATION_TIME_MEASUREMENT
    cudaEventDestroy(m_cuEvtStart);
    cudaEventDestroy(m_cuEvtStop);
    if (EVALTIME_PRINT_LOG == 1)
        m_timeLog.close();
#endif
#ifdef ENABLE_DEBUG_ARRAY
    cudaFreeHost(mp_h_debug);
    cudaFree(mp_d_debug);
#endif
}

errno_t SGE_Evaluator::evaluateDo(
    bool useAbsRef,
    bool calcWF,
    bool cpyGradToCPU,
    bool cpyWfToCPU,
    bool cpyIntToCPU)
{
    // Measure the evaluation time
    // (if ENABLE_EVALUATION_TIME_MEASUREMENT is defined)
    startRecordingTime();

    // Make sure the GPU copy of the darkframe is up to date!
    mp_IHdark->updateGPUCopy();

    // Set reference type
    mp_refManager->setUseAbsReference(useAbsRef);

    // Kernel call - evaluation happens here!
    evaluateSpots<<<mp_GridLayout->mNumSubapertures,
                    mp_GridLayout->mBlockSize,
                    mp_GridLayout->mShmSize>>>(
            mp_IHcam->getWriteBuffer(),              //uint16_t* h_imageData,
            mp_IHdark->getGPUCopy(),                 //float* d_darkData,
            mp_IHcam->mWidth,                        //int imW,
            mp_GridLayout->getDeviceCopy(),          //SGE_GridLayout* d_GridLayout,
            mp_refManager->getSearchPosXGPU(),       //int windowCentersX,
            mp_refManager->getSearchPosYGPU(),       //int windowCentersY,
            mp_refManager->getKernelBufferGPU(),     //float* d_kernel,
            mp_refManager->getRefXGPU(),             //float* d_refX
            mp_refManager->getRefYGPU(),             //float* d_refY
            mp_refManager->getShiftToGradConstant(), //float shift2gradConst
            (float) mp_refManager->getPixelPitch()/2,//float outOfRangeDistance
            mp_IHgradient->getGPUCopy(),             //float* d_gradOut
            mp_IHintensity->getGPUCopy(),            //float* d_intensityOut
            prepareDebugImageDevicePtr(),            //float* d_debugImage
            mp_d_debug);                             //float* d_debugBuffer
    // Print error of kernel launch
    printCE(cudaGetLastError());

    if (calcWF)
    { // Reconstruct the WF while everything is still on the GPU
        mp_wfReconstructor->reconstructWavefrontArrGPU_d2d(
            mp_IHgradient->mNumPx,
            mp_IHgradient->getGPUCopy(),
            mp_IHwf->mNumPx,
            mp_IHwf->getGPUCopy());
    }

    // Copy the results into host memory
    if (cpyGradToCPU)
        mp_IHgradient->updateFromGPU();
    if (calcWF && cpyWfToCPU)
        mp_IHwf->updateFromGPU();
    if (cpyIntToCPU)
        mp_IHintensity->updateFromGPU();

    // If any debug flags are defined, collect data from
    // devide memory and make them accessible / print them
    provideDebugOutputAfterEval();

    return RETURN_SUCCESS;
}

void SGE_Evaluator::setupCudaEnvironment(int deviceID)
{
    cudaError err;
    err = cudaSetDevice(deviceID);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printCE(err);
    err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    printCE(err);
}

void SGE_Evaluator::setupReferenceManager(IMAGE* ref, IMAGE* cam, IMAGE* dark)
{
    mp_refManager = SGE_ReferenceManager::makeReferenceManager(
                                            ref,
                                            cam,
                                            dark,
                                            m_streamPrefix);
}

void SGE_Evaluator::adoptInputStreams(std::string camName, std::string darkName)
{
    mp_IHcam = ImageHandler<uint16_t>::newHandlerAdoptImage(camName);
    cudaError_t err = mp_IHcam->mapImForGPUaccess();
    if (err != cudaSuccess)
        throw std::runtime_error(
            "SGE_Evaluator::SGE_Evaluator: Camera image buffer could not be registered, but this evaluation relies on mapped memory."
            );
    mp_IHdark = ImageHandler<float>::newHandlerAdoptImage(darkName);
}

void SGE_Evaluator::setupGridLayout(int deviceID)
{
    mp_GridLayout = SGE_GridLayout::makeGridLayout(
        deviceID, mp_refManager);
}

void SGE_Evaluator::createOutputImages()
{
    std::string gradientImgName = m_streamPrefix;
    gradientImgName.append("gradOut");
    try {   // See if the gradient image already exists. If so: adopt it!
        mp_IHgradient =
            ImageHandler<float>::newHandlerAdoptImage(gradientImgName);
    }
    catch(std::runtime_error)
    {       // Gradient image does not exist yet. Create a new one.
        mp_IHgradient =
            ImageHandler<float>::newHandlerfrmImage(
                gradientImgName,
                mp_refManager->getRefIH()->getImage());
    }
    mp_IHgradient->setPersistent(true);

    // Prapare result images: intensity
    std::string intensityImgName = m_streamPrefix;
    intensityImgName.append("intOut");
    try {   // See if the intensity image already exists. If so: adopt it!
        mp_IHintensity =
            ImageHandler<float>::newHandlerAdoptImage(intensityImgName);
    }
    catch(std::runtime_error)
    {       // Gradient image does not exist yet. Create a new one.
        mp_IHintensity = ImageHandler<float>::newImageHandler(
            intensityImgName, mp_refManager->getNumSpots(), 1);
    }
    mp_IHintensity->setPersistent(true);

    // Prapare result images: wavefront
    std::string wfImgName = m_streamPrefix;
    wfImgName.append("wfOut");
    try {   // See if the wf image already exists. If so: adopt it!
        mp_IHwf =
            ImageHandler<float>::newHandlerAdoptImage(wfImgName);
    }
    catch(std::runtime_error)
    {       // WF image does not exist yet. Create a new one.
        mp_IHwf = ImageHandler<float>::newImageHandler(
            wfImgName, mp_refManager->getNumSpots(), 1);
    }
    mp_IHwf->setPersistent(true);
}

void SGE_Evaluator::setupWFreconstruction()
{
    ModalWFReconstructorBuilder wfRecBuilder(mp_refManager->getMaskIH(), m_streamPrefix);
    mp_wfReconstructor =  wfRecBuilder.getReconstructor();
}

void SGE_Evaluator::initDebugFields()
{
#ifdef ENABLE_EVALUATION_TIME_MEASUREMENT
    // Create cuda events for timing
    printf("Initializing cuda events for evaluation time printing ...\n");
    cudaError_t err = cudaEventCreate(&m_cuEvtStart);
    err = cudaEventCreate(&m_cuEvtStop);
    printCE(err);

    if (EVALTIME_PRINT_LOG == 1)
    {
        std::string fileName = genereateTimeLogFileName();
        m_timeLog.open(fileName.c_str(), std::ios::app);

        if (!m_timeLog.is_open())
            printf("Failed to open the file: %s\n", fileName.c_str());
        m_timeLog << "# Evaluation time in µs\n";
    }
#endif

#ifdef ENABLE_DEBUG_IMAGE
    // Generate the debug image
    printf("Initializing debugging image for visual inspection of evaluation steps ...\n");
    std::string debugImgName = m_streamPrefix;
    debugImgName.append("DEBUG");
    mp_IHdebug = ImageHandler<float>::newImageHandler(
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
#ifdef ENABLE_EVALUATION_TIME_MEASUREMENT
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
#ifdef ENABLE_EVALUATION_TIME_MEASUREMENT
    // Get timing
    cudaError_t err = cudaEventRecord( m_cuEvtStop, 0 );
    printCE(err);
    err = cudaEventSynchronize( m_cuEvtStop );
    printCE(err);
    float time;
    err = cudaEventElapsedTime( &time, m_cuEvtStart, m_cuEvtStop );
    printCE(err);

    if (EVALTIME_PRINT_LOG == 0)
        printf("Time for kernel call: %.3f µs\n", time*1000);
    else if (EVALTIME_PRINT_LOG == 1)
        m_timeLog << std::fixed << std::setprecision(3) << time*1000 << "\n";
    else
        throw std::runtime_error("SGE_Evaluator::provideDebugOutputAfterEval: Invalid value defined for EVALTIME_PRINT_LOG.");
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

std::string SGE_Evaluator::genereateTimeLogFileName() {
    std::ostringstream oss;

    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);

    oss << m_streamPrefix << "" << std::put_time(tm, "%Y-%m-%d_%H-%M-%S") << "_timing.log";
    return oss.str();
}
