#ifndef SGE_EVALUATOR_HPP
#define SGE_EVALUATOR_HPP

#include "../ref_recorder/SGR_ImageHandler.hpp"
#include <errno.h>

class SGE_GridLayout;

// A class for evaluating SHS images on a GPU
class SGE_Evaluator
{
public:
    // Ctor, doing the initialization
    SGE_Evaluator(
        IMAGE* ref,         // Stream holding the reference data
        IMAGE* cam,         // Camera stream
        IMAGE* dark,        // Dark stream
        int deviceID = 0);  // ID of the GPU device
    // Dtor
    ~SGE_Evaluator();

    // Triggers reading the input- and dark stream
    errno_t evaluateDo();
private:
    int m_deviceID;
    IMAGE* mp_im;
    IMAGE* mp_imDark;
    IMAGE m_imDarkGPU;
    uint16_t m_numSpots;
    uint16_t m_kernelSize;
    std::shared_ptr<SGE_GridLayout> mp_GridLayout;

    uint16_t* m_hp_ROIx;
    uint16_t* m_hp_ROIy;
    uint16_t* m_dp_ROIx;
    uint16_t* m_dp_ROIy;
    float* m_dp_kernel;

    int* m_dp_convCoordsX;
    int* m_dp_convCoordsY;

    // Members for debugging
    cudaEvent_t m_cuEvtStart, m_cuEvtStop;  // Events for timing
    IMAGE m_imDebug;    // Debug image in host memory
    IMAGE m_imDebug_GPU;// Debug image in device memory
    int m_debugBufSize; // Size of the debug buffer
    float* m_hp_debug;  // Debug array in host memory
    float* m_dp_debug;  // Debug array in device memory

    void emulateReferenceInput();
    void writeApertureConvolutionCoordToGPU();
    void copyDarkToGPU();
    void initDebugFields();
    float* copyDebugImgToGPU();
    float* copyDebugImgFrmGPU();
};

#endif // SGR_RECORDER_HPP
