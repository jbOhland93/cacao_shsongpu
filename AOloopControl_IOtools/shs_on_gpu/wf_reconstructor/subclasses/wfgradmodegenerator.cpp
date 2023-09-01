#include "wfgradmodegenerator.hpp"
#include <iostream>

spWGModeGen WFGradModeGenerator::makeWFGradModeGenerator(std::vector<std::pair<spWF, spWFGrad>> samples, double singularValThreshRel)
{
    return spWGModeGen(new WFGradModeGenerator(samples, singularValThreshRel));
}

WFGradModeGenerator::~WFGradModeGenerator()
{
    freeMat(&mMat_SPS_WF_1);
    freeMat(&mMat_SPS_WF_2);
    freeMat(&mMat_SPS_Grd);
    freeVec(&mVec_SPS_WFSVD_S);
    freeMat(&mMat_SPS_WFSVD_V);
    freeVec(&mVec_SPS_WFSVD_work);
    freeMat(&mMat_SPS2WMS);
    freeMat(&mMat_WMS_WF);
    freeMat(&mMat_WMS_Grd);
    freeMat(&mMat_WMS_WF_T);
    freeMat(&mMat_WMS_Grd_T_1);
    freeMat(&mMat_WMS_Grd_T_2);
    freeVec(&mVec_WMS_GrdSVD_S);
    freeMat(&mMat_WMS_GrdSVD_V);
    freeVec(&mVec_WMS_GrdSVD_work);
    freeMat(&mMat_WMS2GMS);
    freeMat(&mMat_GMS_WF);
    freeMat(&mMat_GMS_Grd);
}

std::vector<std::pair<spWF, spWFGrad>> WFGradModeGenerator::calculateModes(int numModes)
{
    // This function is a long and linear series of computations (pun not intended).
    // It is divided into smaller functions which represent logic units for clarity.
    fillSampleMatrices();
    computeSPS2WMStransferMatrix(numModes);
    fillWavefrontModeMatrices();
    computeWMS2GMStransferMatrix();
    fillGradientModeMatrices();
    extractModePairsFromMatrices();

    return getWFModes();
}

WFGradModeGenerator::WFGradModeGenerator(std::vector<std::pair<spWF, spWFGrad>> samples, double singularValThreshRel)
    : mSamples(samples), mSingularValThreshRel(singularValThreshRel)
{
}

void WFGradModeGenerator::freeVec(gsl_vector** vec)
{
    if (*vec != nullptr)
        gsl_vector_free(*vec);
    *vec = nullptr;
}

void WFGradModeGenerator::freeMat(gsl_matrix** mat)
{
    if (*mat != nullptr)
        gsl_matrix_free(*mat);
    *mat = nullptr;
}

void WFGradModeGenerator::initVec(gsl_vector** vec, int size)
{
    freeVec(vec);
    *vec = gsl_vector_alloc(size);
}

void WFGradModeGenerator::initMat(gsl_matrix** mat, int size1, int size2)
{
    freeMat(mat);
    *mat = gsl_matrix_alloc(size1, size2);
}

void WFGradModeGenerator::fillSampleMatrices()
{
    // Fill in the WF sample matrix
    // Determine the number of wavefronts and their size
    int numWavefronts = mSamples.size();
    int wavefrontSize;
    mSamples[0].first->getDataPtr(&wavefrontSize); // Assumes all wavefronts have the same size
    
    // Allocate the matrix
    initMat(&mMat_SPS_WF_1, numWavefronts, wavefrontSize);
    initMat(&mMat_SPS_WF_2, numWavefronts, wavefrontSize);

    // Copy the wavefront data into the matrix
    for (int i = 0; i < numWavefronts; i++)
    {
        double* wavefrontData = mSamples[i].first->getDataPtr(&wavefrontSize);
        for (int j = 0; j < wavefrontSize; j++)
        {
            gsl_matrix_set(mMat_SPS_WF_1, i, j, wavefrontData[j]);
            gsl_matrix_set(mMat_SPS_WF_2, i, j, wavefrontData[j]);
        }
    }

    // Fill in the Grd sample matrix
    // Determine the number of gradients and their size
    int numGradients = mSamples.size();
    int gradientSize;
    mSamples[0].second->getDataPtr(&gradientSize); // Assumes all gradients have the same size
    
    // Allocate the matrix
    initMat(&mMat_SPS_Grd, numGradients, gradientSize);

    // Copy the gradient data into the matrix
    for (int i = 0; i < numGradients; i++)
    {
        double* gradientData = mSamples[i].second->getDataPtr(&gradientSize);
        for (int j = 0; j < gradientSize; j++)
            gsl_matrix_set(mMat_SPS_Grd, i, j, gradientData[j]);
    }
}

void WFGradModeGenerator::computeSPS2WMStransferMatrix(int numModes)
{
    // Calculate the SPS WF SVD
    // Allocate the necessary data structures for the SVD
    initVec(&mVec_SPS_WFSVD_S, mMat_SPS_WF_1->size2);
    initMat(&mMat_SPS_WFSVD_V, mMat_SPS_WF_1->size2, mMat_SPS_WF_1->size2);
    initVec(&mVec_SPS_WFSVD_work, mMat_SPS_WF_1->size2);
    // Compute the SVD
    gsl_linalg_SV_decomp(mMat_SPS_WF_1, mMat_SPS_WFSVD_V, mVec_SPS_WFSVD_S, mVec_SPS_WFSVD_work);

    // Initialize the transfer matrix
    int transferMatS1 = std::min((int) mMat_SPS_WF_1->size2-1, numModes);
    if (transferMatS1 <= 0)
        transferMatS1 = mMat_SPS_WF_1->size2-1;
    initMat(&mMat_SPS2WMS, transferMatS1, mMat_SPS_WF_1->size1); // Skip the last mode, it corresponds to piston

    // Fill in the transfer matrix
    double s, sRec;
    double maxS = gsl_vector_max(mVec_SPS_WFSVD_S);
    for (size_t i = 0; i < mMat_SPS2WMS->size1; i++)
    {
        s = gsl_vector_get(mVec_SPS_WFSVD_S, i);
        sRec = s < mSingularValThreshRel * maxS ? 0 : 1/s; // Inverse of the singular value, but 0 if below threshold
        for (size_t j = 0; j < mMat_SPS2WMS->size2; j++)
            gsl_matrix_set(mMat_SPS2WMS, i, j, sRec * gsl_matrix_get(mMat_SPS_WF_1, j, i));
    }
}

void WFGradModeGenerator::fillWavefrontModeMatrices()
{
    // Transfer the samples to wavefront mode space
    // Initialize the matrices that shall be filled
    initMat(&mMat_WMS_WF, mMat_SPS2WMS->size1, mMat_SPS_WF_1->size2);
    initMat(&mMat_WMS_Grd, mMat_SPS2WMS->size1, mMat_SPS_Grd->size2);
    // The transfer is done by a matrix multiplication with the transfer matrix mMat_SPS2WMS
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mMat_SPS2WMS, mMat_SPS_WF_2, 0.0, mMat_WMS_WF);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mMat_SPS2WMS, mMat_SPS_Grd, 0.0, mMat_WMS_Grd);

    // Transpose the matrices.
    // This is needed for the SVD due to the m>n limitation of the GSL implementation.
    // Transpose WFs:
    initMat(&mMat_WMS_WF_T, mMat_SPS_WF_1->size2, mMat_SPS2WMS->size1);
    for (size_t i = 0; i < mMat_WMS_WF->size1; i++)
        for (size_t j = 0; j < mMat_WMS_WF->size2; j++)
            gsl_matrix_set(mMat_WMS_WF_T, j, i, gsl_matrix_get(mMat_WMS_WF, i, j));
    // Transpose Grads:
    initMat(&mMat_WMS_Grd_T_1, mMat_SPS_Grd->size2, mMat_SPS2WMS->size1);
    initMat(&mMat_WMS_Grd_T_2, mMat_SPS_Grd->size2, mMat_SPS2WMS->size1);
    for (size_t i = 0; i < mMat_WMS_Grd->size1; i++)
        for (size_t j = 0; j < mMat_WMS_Grd->size2; j++)
        {
            gsl_matrix_set(mMat_WMS_Grd_T_1, j, i, gsl_matrix_get(mMat_WMS_Grd, i, j));
            gsl_matrix_set(mMat_WMS_Grd_T_2, j, i, gsl_matrix_get(mMat_WMS_Grd, i, j));
        }
}

void WFGradModeGenerator::computeWMS2GMStransferMatrix()
{
    // Calculate the WMS Gradient SVD
    // Allocate the necessary data structures for the SVD
    initVec(&mVec_WMS_GrdSVD_S, mMat_WMS_Grd_T_1->size2);
    initMat(&mMat_WMS_GrdSVD_V, mMat_WMS_Grd_T_1->size2, mMat_WMS_Grd_T_1->size2);
    initVec(&mVec_WMS_GrdSVD_work, mMat_WMS_Grd_T_1->size2);
    // Compute the SVD
    gsl_linalg_SV_decomp(mMat_WMS_Grd_T_1, mMat_WMS_GrdSVD_V, mVec_WMS_GrdSVD_S, mVec_WMS_GrdSVD_work);

    // Fill in the transfer matrix
    initMat(&mMat_WMS2GMS, mVec_WMS_GrdSVD_S->size, mVec_WMS_GrdSVD_S->size);
    double s, sRec;
    double maxS = gsl_vector_max(mVec_SPS_WFSVD_S);
    for (size_t i = 0; i < mVec_WMS_GrdSVD_S->size; i++)
    {
        s = gsl_vector_get(mVec_WMS_GrdSVD_S, i);
        sRec = s < mSingularValThreshRel * maxS ? 0 : 1/s; // Inverse of the singular value, but 0 if below threshold
        for (size_t j = 0; j < mVec_WMS_GrdSVD_S->size; j++)
            gsl_matrix_set(mMat_WMS2GMS, j, i, sRec * gsl_matrix_get(mMat_WMS_GrdSVD_V, j, i));
    }
}

void WFGradModeGenerator::fillGradientModeMatrices()
{
    // Transfer the samples to gradient mode space
    initMat(&mMat_GMS_WF, mMat_SPS_WF_1->size2, mMat_WMS2GMS->size1);
    initMat(&mMat_GMS_Grd, mMat_SPS_Grd->size2, mMat_WMS2GMS->size1);
    // The transfer is done by a matrix multiplication with the transfer matrix mMat_WMS2GMS
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mMat_WMS_WF_T, mMat_WMS2GMS, 0.0, mMat_GMS_WF);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, mMat_WMS_Grd_T_2, mMat_WMS2GMS, 0.0, mMat_GMS_Grd);
}

void WFGradModeGenerator::extractModePairsFromMatrices()
{
    // Reset modes vector
    mModes.clear();

    // Get the pupil object from the samples
    spPupil pupil = mSamples.at(0).first->getPupil();

    // Each column of mMat_GMS_Grd is a gradient mode and each column of mMat_GMS_WF is the corresponding WF.
    // Convert each column into a wavefront object and add it to mGradientModes
    int gradSize;
    int wfSize;
    for (size_t i = 0; i < mMat_GMS_Grd->size2; i++) {
        // Create a new gradient object, using the pupil object, and get its data pointer
        spWFGrad grdMode = WFGrad::makeWFGrad(pupil);
        double* dataPtrGrd = grdMode->getDataPtr(&gradSize);

        // Copy the data from the i-th column of mMat_GMS_Grd into the gradient
        for (int j = 0; j < gradSize; j++)
            dataPtrGrd[j] = gsl_matrix_get(mMat_GMS_Grd, j, i);

        // Create a new wavefront object, using the pupil object, and get its data pointer
        spWF grdModeWF = Wavefront::makeWavefront(pupil);
        double* dataPtrWF = grdModeWF->getDataPtr(&wfSize);

        // Copy the data from the i-th column of mMat_GMS_WF into the wavefront
        for (int j = 0; j < wfSize; j++)
            dataPtrWF[j] = gsl_matrix_get(mMat_GMS_WF, j, i);

        // Add the mode to mGradientModes
        mModes.push_back(std::pair<spWF, spWFGrad>(grdModeWF, grdMode));
    }
}
