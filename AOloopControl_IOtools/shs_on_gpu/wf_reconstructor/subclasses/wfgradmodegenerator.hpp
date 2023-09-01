#ifndef WFGRADMODEGENERATOR_HPP
#define WFGRADMODEGENERATOR_HPP

#include "wfresponsesampler.hpp"
#include <gsl/gsl_linalg.h>

#define spWGModeGen std::shared_ptr<WFGradModeGenerator>
// A class to calculate the modes of a set of gradients and the associated WFs
class WFGradModeGenerator {
public:
    static spWGModeGen makeWFGradModeGenerator(std::vector<std::pair<spWF, spWFGrad>> samples, double singularValThreshRel = 1e-6);
    ~WFGradModeGenerator();

    // Calculates the modes based on the stored samples.
    // This performs a sequential call of helper functions.
    // Includes all modes (except for the piston) for numModes = -1
    std::vector<std::pair<spWF, spWFGrad>> calculateModes(int numModes = -1);

    int getNumWFSamples() {return mSamples.size(); }
    int getNumModes() { return mModes.size(); }
    int getMaxNumModes() { return mSamples[0].first->getPupil()->getNumValidFields()-1; }
    std::pair<spWF, spWFGrad> getWFSample(int index) { return mSamples.at(index); }
    std::vector<std::pair<spWF, spWFGrad>> getWFSamples() { return mSamples; }
    std::pair<spWF, spWFGrad> getWFMode(int index) { return mModes.at(index); }
    std::vector<std::pair<spWF, spWFGrad>> getWFModes() { return mModes; }



private:
    double mSingularValThreshRel = 1e-6;
    // The WF/Gradient pairs, sampling the pupil
    std::vector<std::pair<spWF, spWFGrad>> mSamples;
    // WF/Gradient paris, representing the pupil modes where the gradients are orthogonormal
    std::vector<std::pair<spWF, spWFGrad>> mModes;

    // =================================================================================
    // Fields for linear algebra
    // =================================================================================
    // SPS - Sample Space
    gsl_matrix* mMat_SPS_WF_1 = nullptr;        // Matrix holding the WF samples - will be overwritten during SVD
    gsl_matrix* mMat_SPS_WF_2 = nullptr;        // Matrix holding the WF samples
    gsl_matrix* mMat_SPS_Grd = nullptr;         // Matrix holding the Grad samples
    gsl_vector* mVec_SPS_WFSVD_S = nullptr;     // Vector for the singular values of the SVD
    gsl_matrix* mMat_SPS_WFSVD_V = nullptr;     // Matrix for the SVD
    gsl_vector* mVec_SPS_WFSVD_work = nullptr;  // Working vector for the SVD implementation

    gsl_matrix* mMat_SPS2WMS = nullptr;         // Transfer matrix from SPS to WMS

    // WMS - Wavefront Mode Space
    gsl_matrix* mMat_WMS_WF = nullptr;          // Matrix where the rows are WF modes
    gsl_matrix* mMat_WMS_Grd = nullptr;         // Matrix where the rows are the gradients of the WF modes
    gsl_matrix* mMat_WMS_WF_T = nullptr;        // Matrix where the columns are WF modes
    gsl_matrix* mMat_WMS_Grd_T_1 = nullptr;     // Matrix where the columns are the gradients of the WF modes - will be overwritten during SVD
    gsl_matrix* mMat_WMS_Grd_T_2 = nullptr;     // Matrix where the columns are the gradients of the WF modes
    gsl_vector* mVec_WMS_GrdSVD_S = nullptr;    // Vector for the singular values of the SVD
    gsl_matrix* mMat_WMS_GrdSVD_V = nullptr;    // Matrix for the SVD
    gsl_vector* mVec_WMS_GrdSVD_work = nullptr; // Working vector for the SVD implementation

    gsl_matrix* mMat_WMS2GMS = nullptr;         // Transfer matrix from WMS to GMS

    // GMS - Gradient Mode Space
    gsl_matrix* mMat_GMS_WF = nullptr;          // Matrix holding the WFs of the gradient modes
    gsl_matrix* mMat_GMS_Grd = nullptr;         // Matrix holding the gradient modes
    // =================================================================================
    
    WFGradModeGenerator(); // No publicly available constructor
    WFGradModeGenerator(std::vector<std::pair<spWF, spWFGrad>> samples, double singularValThreshRel); // Private constructor
    
    // =================================================================================
    // Helper functions
    // =================================================================================
    // Allocation and deallocation of matrices and vectors for readibility
    void freeVec(gsl_vector** vec);
    void freeMat(gsl_matrix** mat);
    void initVec(gsl_vector** vec, int size);
    void initMat(gsl_matrix** mat, int size1, int size2);

    // Takes the response pair samples and writes their values into the SPS sample matrices:
    // mMat_SPS_WF_1
    // mMat_SPS_WF_2
    // mMat_SPS_Grd
    void fillSampleMatrices();

    // Performs a SVD on mMat_SPS_WF_1 and computes the transfer from sample space
    // into wavefront mode space, which is written into mMat_SPS2WMS
    void computeSPS2WMStransferMatrix(int numModes = -1);

    // Takes the mMat_SPS2WMS transfer matrix and projects the WFs and Gradients
    // into wavefront mode space.
    void fillWavefrontModeMatrices();

    // Performs a SVD on mMat_WMS_Grd_T_1 and computes the transfer from wavefront mode space
    // into gradient mode space, which is written into mMat_WMS2GMS
    void computeWMS2GMStransferMatrix();

    // Takes the mMat_WMS2GMS transfer matrix and projects the WFs and Gradients
    // into gradient mode space.
    void fillGradientModeMatrices();

    // Takes the mMat_GMS_WF and mMat_GMS_Grd matrices in gradient mode space, extracts the
    // wavefront/gradient pairs and writes them into mModes.
    void extractModePairsFromMatrices();

};

#endif // WFGRADMODEGENERATOR_HPP