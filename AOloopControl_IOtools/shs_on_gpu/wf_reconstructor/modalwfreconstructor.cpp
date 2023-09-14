#include "modalwfreconstructor.hpp"
#include <iostream>
#include <cstring>
#include <cuda.h>
#include "../util/CudaUtil.hpp"

spWFReconst ModalWFReconstructor::makeWFReconstructor(
    std::vector<std::pair<spWF, spWFGrad>> modes,
    std::string streamPrefix)
{
    return spWFReconst(new ModalWFReconstructor(modes, streamPrefix));
}

ModalWFReconstructor::~ModalWFReconstructor()
{
    cublasDestroy(m_cublasHandle);
    if (mp_d_gradient != nullptr)
        cudaFree(mp_d_gradient);
    if (mp_d_wf != nullptr)
        cudaFree(mp_d_wf);
}

void ModalWFReconstructor::reconstructWavefrontArrCPU(int gradientLength, float* h_gradientArr, int wfLength, float* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);

    // Get vector views
    gsl_vector_float_view gradientView =
        gsl_vector_float_view_array(h_gradientArr, gradientLength);
    gsl_vector_float_view wfView =
        gsl_vector_float_view_array(h_wfArrDst, wfLength);

    // Multiply gradient with the reconstruction matrix
    gsl_blas_sgemv(CblasNoTrans, 1.0,
        mp_reconstMat,
        &gradientView.vector,
        0.0,
        &wfView.vector);
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_d2d(int gradientLength, float* d_gradientArr, int wfLength, float* d_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);

    // Get the pointer to the GPU copy of the reconstruction matrix
    float* d_reconstMat = mp_IHreconstructionMatrix->getGPUCopy();

    // Perform the matrix-vector multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t err = cublasSgemv(
        m_cublasHandle,
        CUBLAS_OP_N,
        wfLength,       // rows, equal to wf length
        gradientLength, // columns, equal to grad length
        &alpha,         
        d_reconstMat,   // device copy of reconst mat
        wfLength,       // rows, equals wf length
        d_gradientArr,  // gradient vector
        1,              // increments x
        &beta,
        d_wfArrDst,     // wf vector
        1);             // increments y
    printCuBE(err);
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_h2d(int gradientLength, float* h_gradientArr, int wfLength, float* d_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    // Copy the gradient to the pre-allocated gradient array on the device
    printCE(cudaMemcpy(mp_d_gradient, h_gradientArr, gradientLength*sizeof(float), cudaMemcpyHostToDevice));
    // Perform the reconstruction on the gpu
    reconstructWavefrontArrGPU_d2d(gradientLength, mp_d_gradient, wfLength, d_wfArrDst);
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_d2h(int gradientLength, float* d_gradientArr, int wfLength, float* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    // Perform the reconstruction on the gpu
    reconstructWavefrontArrGPU_d2d(gradientLength, d_gradientArr, wfLength, mp_d_wf);
    // Copy the resulting wf to the host destination
    printCE(cudaMemcpy(h_wfArrDst, mp_d_wf, wfLength*sizeof(float), cudaMemcpyDeviceToHost));
}

void ModalWFReconstructor::reconstructWavefrontArrGPU_h2h(int gradientLength, float* h_gradientArr, int wfLength, float* h_wfArrDst)
{
    checkArraySizes(gradientLength, wfLength);
    // Copy the gradient to the pre-allocated gradient array on the device
    printCE(cudaMemcpy(mp_d_gradient, h_gradientArr, gradientLength*sizeof(float), cudaMemcpyHostToDevice));
    // Perform a device-to-host reconstruction, using the copy of the gradient
    reconstructWavefrontArrGPU_d2h(gradientLength, mp_d_gradient, wfLength, h_wfArrDst);
}

ModalWFReconstructor::ModalWFReconstructor(
    std::vector<std::pair<spWF, spWFGrad>> modes,
    std::string streamPrefix)
{
    cublasStatus_t CuBErr = cublasCreate(&m_cublasHandle);
    printCuBE(CuBErr);
    mPupil = modes[0].first->getPupil();
    setupReconstructionMatrix(modes, streamPrefix);
    // Pre-allocate device vectors for a gradient and wf for faster execution
    cudaError_t err;
    err = cudaMalloc((void**) &mp_d_gradient, mPupil->getNumValidFields()*2*sizeof(float));
    printCE(err);
    err = cudaMalloc((void**) &mp_d_wf, mPupil->getNumValidFields()*sizeof(float));
    printCE(err);
}

void ModalWFReconstructor::setupReconstructionMatrix(
    std::vector<std::pair<spWF, spWFGrad>> modes,
    std::string streamPrefix)
{
    int numModes = modes.size();
    int wfSize, gradSize;
    double* wfData = modes[0].first->getDataPtr(&wfSize);
    double* gradData = modes[0].second->getDataPtr(&gradSize);

    // Allocate the reconstruction matrix
    gsl_matrix* rmDbl = gsl_matrix_alloc(wfSize, gradSize);

    // For each mode, convert WF and gradient to gsl_vector and multiply them into the reconstruction matrix
    for (int i = 0; i < numModes; ++i)
    {
        wfData = modes[i].first->getDataPtr(&wfSize);
        gradData = modes[i].second->getDataPtr(&gradSize);
        gsl_vector_view wfView = gsl_vector_view_array(wfData, wfSize);
        gsl_vector_view gradView = gsl_vector_view_array(gradData, gradSize);

        gsl_blas_dger(1.0, &wfView.vector, &gradView.vector, rmDbl);
    }

    // Port the matrix to an image stream of type float
    std::string imgName = streamPrefix;
    imgName.append("ReconstMat");
    mp_IHreconstructionMatrix = ImageHandler<float>::newImageHandler(
        imgName,
        wfSize,
        gradSize);
    float* imgPtr = mp_IHreconstructionMatrix->getWriteBuffer();
    for (int x = 0; x < wfSize; x++)
        for (int y = 0; y < gradSize; y++)
            imgPtr[x + y*wfSize] = (float) gsl_matrix_get(rmDbl, x, y);
    mp_IHreconstructionMatrix->updateWrittenImage();

    // Create the float reconstruction matrix out of the image
    m_reconstMatView = gsl_matrix_float_view_array(imgPtr, wfSize, gradSize);
    mp_reconstMat = &m_reconstMatView.matrix;
}

void ModalWFReconstructor::checkArraySizes(int gradientLength, int wfLength)
{
    // Check the array sizes
    if (gradientLength != mp_reconstMat->size2)
        throw std::runtime_error("ModalWFReconstructor::checkArraySizes: the gradient length does not match the reconstruction matrix size.");
    if (wfLength != mp_reconstMat->size1)
        throw std::runtime_error("ModalWFReconstructor::checkArraySizes: the wf length does not match the reconstruction matrix size.");
}