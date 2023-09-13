#ifndef SGE_EVALUATOR_HPP
#define SGE_EVALUATOR_HPP

#include "SGE_ReferenceManager.hpp"
#include "SGE_GridLayout.hpp"
#include "../util/ImageHandler.hpp"
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
        IMAGE* ref,                 // Stream holding the reference data
        IMAGE* cam,                 // Camera stream
        IMAGE* dark,                // Dark stream
        const char* streamPrefix,   // Prefix for the ISIO streams
        int deviceID = 0);          // ID of the GPU device
    // Dtor
    ~SGE_Evaluator();

    // Triggers the evaluation
    errno_t evaluateDo(bool useAbsoluteReference);
private:
    std::string m_streamPrefix;

    spImageHandler(uint16_t) mp_IHcam;
    spImageHandler(float) mp_IHdark;
    spRefManager mp_refManager;
    spGridLayout mp_GridLayout;

    // device copy of the spot reference positions X
    float* mp_d_refX;
    // device copy of the spot reference positions Y
    float* mp_d_refY;
    // device ptr to the current integer spot searching positions X
    uint16_t* mp_d_SearchPosX;
    // device ptr to the current integer spot searching positions Y
    uint16_t* mp_d_SearchPosY;

    // The image holding the WF gradient after the image eval
    spImageHandler(float) mp_IHgradient;
    // The modal WF reconstructor on the pupil of the reference
    spWFReconst mp_wfReconstructor;
    // The image holding the reconstructed WF
    spImageHandler(float) mp_IHwf;

    // Members for debugging
    cudaEvent_t m_cuEvtStart, m_cuEvtStop;  // Events for timing
    spImageHandler(float) mp_IHdebug = nullptr; // Debug image in host memory
    int m_debugBufSize; // Size of the debug buffer
    float* mp_h_debug;  // Debug array in host memory
    float* mp_d_debug;  // Debug array in device memory
    std::ofstream m_timeLog; // Ouput stream for time logging

    // Debugging functions
    void initDebugFields();
    void startRecordingTime();
    float* prepareDebugImageDevicePtr();
    void provideDebugOutputAfterEval();
    std::string genereateTimeLogFileName();
};

#endif // SGE_EVALUATOR_HPP
