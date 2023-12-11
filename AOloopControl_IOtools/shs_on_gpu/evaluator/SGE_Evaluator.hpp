#ifndef SGE_EVALUATOR_HPP
#define SGE_EVALUATOR_HPP

#include "SGE_ReferenceManager.hpp"
#include "SGE_GridLayout.hpp"
#include "../util/ImageHandler2D.hpp"
#include "../wf_reconstructor/modalwfreconstructor.hpp"
#include <errno.h>
#include <iostream>
#include <fstream>

// A class for evaluating SHS images on a GPU
class SGE_Evaluator
{
public:
    // Ctor, doing the initialization
    SGE_Evaluator(
        FUNCTION_PARAMETER_STRUCT* fps, // process related fps
        IMAGE* ref,                 // Stream holding the reference data
        IMAGE* cam,                 // Camera stream
        IMAGE* dark,                // Dark stream
        const char* streamPrefix,   // Prefix for the ISIO streams
        int deviceID = 0);          // ID of the GPU device
    // Dtor
    ~SGE_Evaluator();

    // Triggers the evaluation
    errno_t evaluateDo(
        bool useAbsRef,     // Reference will be absolute w.r.t. the MLA grid
        bool calcWF,        // Calculate the WF from the gradient field
        bool cpyGradToCPU,  // Copy the evaluated gradient to the CPU
        bool cpyWfToCPU,    // Copy the WF to the CPU, if reconstructed
        bool cpyIntToCPU,   // Copy the intensity to the CPU
        bool logWFstats);   // Log wf PtV and RMS to file
private:
    FUNCTION_PARAMETER_STRUCT* mp_fps;

    std::string m_streamPrefix;

    spImHandler2D(uint16_t) mp_IHcam;
    spImHandler2D(float) mp_IHdark;
    spRefManager mp_refManager;
    spGridLayout mp_GridLayout;

    // The image holding the WF gradient after the image eval
    spImHandler2D(float) mp_IHgradient;
    // The image holding the intensity over the pupil
    spImHandler2D(float) mp_IHintensity;
    // The modal WF reconstructor on the pupil of the reference
    spWFReconst mp_wfReconstructor;
    // The image holding the reconstructed WF
    spImHandler2D(float) mp_IHwf;

    // Setup functions
    void setupCudaEnvironment(int deviceID);
    void setupReferenceManager(IMAGE* ref, IMAGE* cam, IMAGE* dark);
    void adoptInputStreams(std::string camName, std::string darkName);
    void setupGridLayout(int deviceID);
    void createOutputImages();
    void setupWFreconstruction();

    // Misc
    void logWFstatsToFile(bool doLog);
    std::ofstream m_wfStatsLog;

    // Members for debugging
    cudaEvent_t m_cuEvtStart, m_cuEvtStop;  // Events for timing
    spImHandler2D(float) mp_IHdebug = nullptr; // Debug image in host memory
    int m_debugBufSize; // Size of the debug buffer
    float* mp_h_debug;  // Debug array in host memory
    float* mp_d_debug;  // Debug array in device memory
    std::ofstream m_timeLog; // Ouput stream for time logging

    // Debugging functions
    void initDebugFields();
    void startRecordingTime();
    float* prepareDebugImageDevicePtr();
    void provideDebugOutputAfterEval();
    std::string genereateLogFileName(std::string suffix);
};

#endif // SGE_EVALUATOR_HPP
