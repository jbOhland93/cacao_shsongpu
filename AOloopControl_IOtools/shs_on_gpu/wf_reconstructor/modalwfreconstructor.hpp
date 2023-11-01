#ifndef MODALWFRECONSTRUCTOR_HPP
#define MODALWFRECONSTRUCTOR_HPP

#include "subclasses/wfgradmodegenerator.hpp"
#include "../util/ImageHandler2D.hpp"
#include <cublas_v2.h>

#define spWFReconst std::shared_ptr<ModalWFReconstructor>
// A class setting up and providing a WF reconstruction mechanism
class ModalWFReconstructor {
public:
    static spWFReconst makeWFReconstructor(
        std::vector<std::pair<spWF, spWFGrad>> modes,
        std::string streamPrefix);
    ~ModalWFReconstructor();

    // Reconstructs a wavefront completely on the host system, using the array representation of WFs and gradients
    void reconstructWavefrontArrCPU(int gradientLength, float* h_gradientArr, int wfLength, float* h_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source and destination arrays reside on the device
    void reconstructWavefrontArrGPU_d2d(int gradientLength, float* d_gradientArr, int wfLength, float* d_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source array resides in host memory and the destination on the device
    void reconstructWavefrontArrGPU_h2d(int gradientLength, float* h_gradientArr, int wfLength, float* d_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source array resides on the device and the destination in host memory
    void reconstructWavefrontArrGPU_d2h(int gradientLength, float* d_gradientArr, int wfLength, float* h_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source and destination arrays reside in host memory
    void reconstructWavefrontArrGPU_h2h(int gradientLength, float* h_gradientArr, int wfLength, float* h_wfArrDst);

private:
    cublasHandle_t m_cublasHandle;
    spPupil mPupil;
    spImHandler2D(float) mp_IHreconstructionMatrix;
    gsl_matrix_float_view m_reconstMatView;
    gsl_matrix_float* mp_reconstMat;

    // Pre-allocated device array for gradient vectors
    float* mp_d_gradient = nullptr;
    // Pre-allocated device array for wavefronts
    float* mp_d_wf = nullptr;

    ModalWFReconstructor(); // No publicly available constructor
     // Private constructor
    ModalWFReconstructor(
        std::vector<std::pair<spWF, spWFGrad>> modes,
        std::string streamPrefix);

    void setupReconstructionMatrix(
        std::vector<std::pair<spWF, spWFGrad>> modes,
        std::string streamPrefix);
    void checkArraySizes(int gradientLength, int wfLength);
};

#endif // MODALWFRECONSTRUCTOR_HPP
