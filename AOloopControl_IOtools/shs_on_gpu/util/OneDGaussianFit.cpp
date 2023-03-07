#include "OneDGaussianFit.hpp"

#include <algorithm>

using namespace OneDGaussianFit;

// A data structure for the solver
struct fitData
{
    size_t size;
    double * values;
    double * weights;
};

void OneDGaussianFit::fitGaussian(
    std::shared_ptr<double[]> values,
    const size_t dataSize,
    gaussianParams pInit,
    gaussianParams* pOut,
    std::shared_ptr<double[]> recreation)
{
    if (pOut == nullptr)
        throw std::runtime_error(
            "OneDGaussianFit::fitGaussian: pOut cannot be a nullpointer.");

    // Set up fit data struct
    double weights[dataSize];
    std::fill_n(weights, dataSize, 1.);
    double* yValues = values.get();
    struct fitData data = {dataSize, yValues, weights};

    // Initialize the fit parameter vector
    const size_t numFitParams = 4;
    double fitParams[] = {
        pInit.amplitude,
        pInit.mean,
        pInit.stddev,
        pInit.offset};
    gsl_vector_view fitParamVector = gsl_vector_view_array(fitParams, numFitParams);

    // Set up the GLS function for the fit
    gsl_multifit_function_fdf glsFunction;
    glsFunction.f = &gsl_gaussian;
    glsFunction.df = &gsl_gaussJacobian;
    glsFunction.fdf = &gsl_cmpt_gaussAndJac;
    glsFunction.n = dataSize;
    glsFunction.p = numFitParams;
    glsFunction.params = &data;

    // Set up the solver for the fit
    const gsl_multifit_fdfsolver_type *solverType;
    gsl_multifit_fdfsolver *solver;
    solverType = gsl_multifit_fdfsolver_lmsder;
    solver = gsl_multifit_fdfsolver_alloc (solverType, dataSize, numFitParams);
    gsl_multifit_fdfsolver_set (solver, &glsFunction, &fitParamVector.vector);

    // Do fit!
    int status;
    size_t iteration = 0;
    while (status == GSL_CONTINUE && iteration < 500)
    {
        iteration++;
        status = gsl_multifit_fdfsolver_iterate(solver);

        if (status) // Break if an error occured
            break;

        status = gsl_multifit_test_delta(solver->dx, solver->x, 1e-4, 1e-4);
        if (status != GSL_CONTINUE)
            break;
    }

    // Store fit parameters in output
    pOut->amplitude = gsl_vector_get(solver->x, 0);
    pOut->mean = gsl_vector_get(solver->x, 1);
    pOut->stddev = gsl_vector_get(solver->x, 2);
    pOut->offset = gsl_vector_get(solver->x, 3);

    // If the recreated function is required generate it.
    if (recreation != nullptr)
    {
        for (int i = 0; i < dataSize; i++)
            recreation[i] =
                pOut->amplitude *
                exp(-pow(i-pOut->mean,2)/(2*pow(pOut->stddev,2)))
                + pOut->offset;
    }

    gsl_multifit_fdfsolver_free (solver);
}

int OneDGaussianFit::gsl_gaussian(
        const gsl_vector * fitParams,
        void *rawData,
        gsl_vector * computedFunction)
{
    double A = gsl_vector_get(fitParams, 0);       // Parameter: Amplitude
    double mu = gsl_vector_get(fitParams, 1);      // Parameter: Center
    double stdev = gsl_vector_get(fitParams, 2);   // Parameter: Stddeviation
    double off = gsl_vector_get(fitParams, 3);     // Parameter: Offset

    size_t size = ((fitData *)rawData)->size;
    double *values = ((fitData *)rawData)->values;
    double *weights = ((fitData *)rawData)->weights;

    for (size_t i = 0; i < size; i++)
    {
        // Yi holds the evaluation of the current model for each sample
        double Yi = A * exp (- pow(i-mu, 2)/(2*pow(stdev, 2))) + off;
        gsl_vector_set(computedFunction, i, (Yi - values[i])/weights[i]);
    }
    return GSL_SUCCESS; 
}

int OneDGaussianFit::gsl_gaussJacobian(
        const gsl_vector * fitParams,
        void *rawData,
        gsl_matrix * J)
{
    double A = gsl_vector_get(fitParams, 0);       // Parameter: Amplitude
    double mu = gsl_vector_get(fitParams, 1);      // Parameter: Center
    double stdev = gsl_vector_get(fitParams, 2);   // Parameter: Stddeviation
    double off = gsl_vector_get(fitParams, 3);     // Parameter: Offset

    size_t size = ((fitData *)rawData)->size;
    double *weights = ((fitData *)rawData)->weights;

    for (size_t i = 0; i < size; i++)
    {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-(i-mu)^2/(2*stdev^2))) + b  */
        /* and the xj are the parameters (A,mu,stdef,b) */
        double w = weights[i];
        double cntr = i-mu;
        double e = exp(- pow(cntr, 2)/(2*pow(stdev, 2)));
        gsl_matrix_set (J, i, 0, e/w);                              // dfi/dA
        gsl_matrix_set (J, i, 1, A*cntr/pow(stdev,2) * e/w);        // dfi/dmu
        gsl_matrix_set (J, i, 2, A*cntr*cntr/pow(stdev,3) * e/w);   // dfi/dstdev
        gsl_matrix_set (J, i, 3, 1/w);                              // dfi/doff
    }
    return GSL_SUCCESS;
}

int OneDGaussianFit::gsl_cmpt_gaussAndJac(
        const gsl_vector* fitParams,
        void* rawData,
        gsl_vector* computedFunction,
        gsl_matrix* J)
{
    gsl_gaussian(fitParams, rawData, computedFunction);
    gsl_gaussJacobian(fitParams, rawData, J);
    return GSL_SUCCESS;
}
