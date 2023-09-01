#ifndef MODALWFRECONSTRUCTOR_HPP
#define MODALWFRECONSTRUCTOR_HPP

#include "subclasses/wfgradmodegenerator.hpp"

#define spWFReconst std::shared_ptr<ModalWFReconstructor>
// A class setting up and providing a WF reconstruction mechanism
class ModalWFReconstructor {
public:
    static spWFReconst makeWFReconstructor(std::vector<std::pair<spWF, spWFGrad>> modes);
    ~ModalWFReconstructor();

    // Sets the device ID which is used for reconstruction on the device
    void setDeviceID(int deviceID) { mDeviceID = deviceID; }

    // Reconstructs a wavefront completely on the host system, using the class representation of WFs and gradients
    spWF reconstructWavefront(spWFGrad gradient);
    // Reconstructs a wavefront completely on the host system, using the array representation of WFs and gradients
    void reconstructWavefrontArrCPU(int gradientLength, double* h_gradientArr, int wfLength, double* h_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source and destination arrays reside on the device
    void reconstructWavefrontArrGPU_d2d(int gradientLength, double* d_gradientArr, int wfLength, double* d_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source array resides in host memory and the destination on the device
    void reconstructWavefrontArrGPU_h2d(int gradientLength, double* h_gradientArr, int wfLength, double* d_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source array resides on the device and the destination in host memory
    void reconstructWavefrontArrGPU_d2h(int gradientLength, double* d_gradientArr, int wfLength, double* h_wfArrDst);
    // Reconstructs a wavefront on the gpu, where source and destination arrays reside in host memory
    void reconstructWavefrontArrGPU_h2h(int gradientLength, double* h_gradientArr, int wfLength, double* h_wfArrDst);

private:
    int mDeviceID = 0;
    spPupil mPupil;
    gsl_matrix* mReconstructionMatrix;

    ModalWFReconstructor(); // No publicly available constructor
    ModalWFReconstructor(std::vector<std::pair<spWF, spWFGrad>> modes); // Private constructor

    void setupReconstructionMatrix(std::vector<std::pair<spWF, spWFGrad>> modes);
    void checkArraySizes(int gradientLength, int wfLength);
};

#endif // MODALWFRECONSTRUCTOR_HPP
