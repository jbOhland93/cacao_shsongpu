#include "modalwfreconstructor.hpp"
#include <iostream>
#include <cstring>

spWFReconst ModalWFReconstructor::makeWFReconstructor(std::vector<std::pair<spWF, spWFGrad>> modes)
{
    return spWFReconst(new ModalWFReconstructor(modes));
}

ModalWFReconstructor::~ModalWFReconstructor()
{
    gsl_matrix_free(mReconstructionMatrix);
}

spWF ModalWFReconstructor::reconstructWavefront(spWFGrad gradient)
{
    // Get gradient data
    int gradientSize;
    double* gradientData = gradient->getDataPtr(&gradientSize);

    // Generate the WF object for the reconstruction
    int wfSize = mReconstructionMatrix->size1;
    spWF wavefront = Wavefront::makeWavefront(mPupil);
    double* wfDataPtr = wavefront->getDataPtr(&wfSize);
    
    // Reconstruct
    reconstructWavefrontArrCPU(gradientSize, gradientData, wfSize, wfDataPtr);
    
    return wavefront;
}

void ModalWFReconstructor::reconstructWavefrontArrCPU(int gradientLength, double* h_gradientArr, int wfLength, double* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);

    // Get vector views
    gsl_vector_view gradientView = gsl_vector_view_array(h_gradientArr, gradientLength);
    gsl_vector_view wfView = gsl_vector_view_array(h_wfArrDst, wfLength);

    // Multiply gradient with the reconstruction matrix
    gsl_blas_dgemv(CblasNoTrans, 1.0, mReconstructionMatrix, &gradientView.vector, 0.0, &wfView.vector);
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_d2d(int gradientLength, double* d_gradientArr, int wfLength, double* d_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);

    // Use mDeviceID
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_d2d: ToDo - implement mat mul on GPU!\n");
    throw std::runtime_error("ModalWFReconstructor::reconstructWavefrontArrGPU_d2d: ToDo - implement mat mul on GPU!\n");
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_h2d(int gradientLength, double* h_gradientArr, int wfLength, double* d_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_h2d: ToDo - copy gradient to GPU!\n");
    double* d_gradientArr = nullptr; // Do allocation and copying here - use mDeviceID.
    reconstructWavefrontArrGPU_d2d(gradientLength, d_gradientArr, wfLength, d_wfArrDst);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_h2d: ToDo - Free gradient on GPU!\n");
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_d2h(int gradientLength, double* d_gradientArr, int wfLength, double* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_d2h: ToDo - allocate wf array on GPU!\n");
    double* d_wfArrDst = nullptr; // Do allocation here - use mDeviceID.
    reconstructWavefrontArrGPU_d2d(gradientLength, d_gradientArr, wfLength, d_wfArrDst);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_d2h: ToDo - Copy WF to host, free device array!\n");
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_h2h(int gradientLength, double* h_gradientArr, int wfLength, double* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_h2h: ToDo - allocate grad & wf array on GPU, copy grad!\n");
    double* d_gradientArr = nullptr; // Do allocation and copying here - use mDeviceID.
    double* d_wfArrDst = nullptr; // Do allocation  here - use mDeviceID.
    reconstructWavefrontArrGPU_d2d(gradientLength, d_gradientArr, wfLength, d_wfArrDst);
    printf("ModalWFReconstructor::reconstructWavefrontArrGPU_h2h: ToDo - Copy WF to host, free device arrays!\n");
}

ModalWFReconstructor::ModalWFReconstructor(std::vector<std::pair<spWF, spWFGrad>> modes)
{
    mPupil = modes[0].first->getPupil();
    setupReconstructionMatrix(modes);
}

void ModalWFReconstructor::setupReconstructionMatrix(std::vector<std::pair<spWF, spWFGrad>> modes)
{
    int numModes = modes.size();
    int wfSize, gradSize;
    double* wfData = modes[0].first->getDataPtr(&wfSize);
    double* gradData = modes[0].second->getDataPtr(&gradSize);

    // Allocate the reconstruction matrix
    mReconstructionMatrix = gsl_matrix_alloc(wfSize, gradSize);

    // For each mode, convert WF and gradient to gsl_vector and multiply them into the reconstruction matrix
    for (int i = 0; i < numModes; ++i)
    {
        wfData = modes[i].first->getDataPtr(&wfSize);
        gradData = modes[i].second->getDataPtr(&gradSize);
        gsl_vector_view wfView = gsl_vector_view_array(wfData, wfSize);
        gsl_vector_view gradView = gsl_vector_view_array(gradData, gradSize);

        gsl_blas_dger(1.0, &wfView.vector, &gradView.vector, mReconstructionMatrix);
    }
}

void ModalWFReconstructor::checkArraySizes(int gradientLength, int wfLength)
{
    // Check the array sizes
    if (gradientLength != mReconstructionMatrix->size2)
        throw std::runtime_error("ModalWFReconstructor::checkArraySizes: the gradient length does not match the reconstruction matrix size.");
    if (wfLength != mReconstructionMatrix->size1)
        throw std::runtime_error("ModalWFReconstructor::checkArraySizes: the wf length does not match the reconstruction matrix size.");
}