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
        IMAGE* in,          // Raw camera stream
        IMAGE* dark,        // Stream holding a dark for subtraction
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

    cudaEvent_t m_cuEvtStart, m_cuEvtStop;
    IMAGE m_imDebug;
    IMAGE m_imDebug_GPU;
    int m_debugBufSize;
    float* m_hp_debug;
    float* m_dp_debug;

    void emulateReferenceInput();
    void writeApertureConvolutionCoordToGPU();
    void copyDarkToGPU();
    void initDebugFields();
    float* copyDebugImgToGPU();
    float* copyDebugImgFrmGPU();
};

#endif // SGR_RECORDER_HPP
