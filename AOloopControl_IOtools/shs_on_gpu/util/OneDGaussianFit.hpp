// Fitting utility for a 1D gaussian function to a double array of values
// Uses the GSL library
// Heavily guided by this example:
//  https://www.csse.uwa.edu.au/programming/gsl-1.0/gsl-ref_35.html
#ifndef ONEDGAUSSIANFIT_HPP
#define ONEDGAUSSIANFIT_HPP

#include <memory>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

namespace OneDGaussianFit
{
    // A structure holding the parameters of a 1D gaussian functin
    struct gaussianParams {
        double amplitude;
        double mean;
        double stddev;
        double offset;
    };

    // Fits a 1D gaussian to the given data.
    // values: The array of values to be fitted
    // dataSize: The length of the value array
    // pInit: Initial parameters for the fit
    // pOut: A pointer to the location where the result
    //      should be stored
    // recreation: A pointer to the location where
    //      the values of the fitted function should
    //      be stored. Can be null.
    void fitGaussian(
        std::shared_ptr<double[]> values,
        const size_t dataSize,
        gaussianParams pInit,
        gaussianParams* pOut,
        std::shared_ptr<double[]> recreation = nullptr);

    // Helper function for GSL fitting:
    //      Computes the gaussian
    // fitParams: fit parameters
    // rawData: pointer to the fitData
    // computedFunction: Destination for the computed function
    int gsl_gaussian(
        const gsl_vector * fitParams,
        void *rawData,
        gsl_vector * computedFunction);

    // Helper function for GSL fitting:
    //      Computes the Jacobian matrix of the gaussian
    // fitParams: fit parameters
    // rawData: pointer to the fitData
    // J: Destination for the computed Jacobian matrix
    int gsl_gaussJacobian(
        const gsl_vector * fitParams,
        void *rawData,
        gsl_matrix * J);

    // Helper function for GSL fitting:
    //      Computes the gaussian and its Jacobian matrix
    // fitParams: fit parameters
    // rawData: pointer to the fitData
    // computedFunction: Destination for the computed function
    // J: Destination for the computed Jacobian matrix
    int gsl_cmpt_gaussAndJac(
        const gsl_vector* fitParams,
        void* rawData,
        gsl_vector* computedFunction,
        gsl_matrix* J);
}

#endif // ONEDGAUSSIANFIT_HPP