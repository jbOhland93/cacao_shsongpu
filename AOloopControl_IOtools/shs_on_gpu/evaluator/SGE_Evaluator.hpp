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
    IMAGE m_imDarkGPU;
    std::shared_ptr<SGE_GridLayout> mpGridLayout;

    uint16_t* m_hp_ROIx;
    uint16_t* m_hp_ROIy;
    uint16_t* m_dp_ROIx;
    uint16_t* m_dp_ROIy;
    float* m_dp_kernel;

    void copyDarkToGPU(IMAGE* dark);
};

#endif // SGR_RECORDER_HPP
