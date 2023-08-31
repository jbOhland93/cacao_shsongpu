#ifndef SGE_EVALUATOR_HPP
#define SGE_EVALUATOR_HPP

#include "../ref_recorder/SGR_ImageHandler.hpp"
#include <errno.h>
#include "SGE_ReferenceManager.hpp"

class SGE_GridLayout;

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

    // Triggers reading the input- and dark stream
    errno_t evaluateDo();
private:
    int m_deviceID;
    std::string m_streamPrefix;

    spImageHandler(uint16_t) mp_IHcam;
    spImageHandler(float) mp_IHdark;
    spRefManager mp_refManager;
    std::shared_ptr<SGE_GridLayout> mp_GridLayout;

    uint16_t* mp_d_SearchPosX;
    uint16_t* mp_d_SearchPosY;

    // Members for debugging
    cudaEvent_t m_cuEvtStart, m_cuEvtStop;  // Events for timing
    spImageHandler(float) mp_IHdebug;    // Debug image in host memory
    int m_debugBufSize; // Size of the debug buffer
    float* mp_h_debug;  // Debug array in host memory
    float* mp_d_debug;  // Debug array in device memory

    void writeApertureConvolutionCoordToGPU();
    void initDebugFields();
};

#endif // SGR_RECORDER_HPP
