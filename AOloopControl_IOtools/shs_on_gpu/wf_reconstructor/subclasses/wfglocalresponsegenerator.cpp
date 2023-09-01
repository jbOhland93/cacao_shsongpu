#include "wfglocalresponsegenerator.hpp"
#include <iostream>
#include <cmath>

spWGLRspnsGen WFGLocalResponseGenerator::makeLocalResponseGenerator(spPupil pupil)
{
    return spWGLRspnsGen(new WFGLocalResponseGenerator(pupil));
}

std::pair<spWF, spWFGrad> WFGLocalResponseGenerator::generateResponse(double centerX, double centerY, double width)
{
    // Initialize WF and Grad objects
    spWF wf = Wavefront::makeWavefront(mPupil);
    spWFGrad grad = WFGrad::makeWFGrad(mPupil);

    // Get data array pointers
    int numSamples;
    bool* pupArr = mPupil->getDataPtr(&numSamples);
    double* wfVals = wf->getDataPtr(&numSamples);
    double* dxVals = grad->getDataPtrDX(&numSamples);
    double* dyVals = grad->getDataPtrDY(&numSamples);

    // Get width and height of the pupil array
    int pupWwidth = mPupil->getWidth();
    int pupHeight = mPupil->getHeight();

    // Loop over all subapertures
    int i = 0;
    for (int y = 0; y < pupHeight; y++)
        for (int x = 0; x < pupWwidth; x++)
            if (pupArr[y*pupWwidth+x])
            {
                wfVals[i] = lorentzian(x, y, centerX, centerY, width);
                std::pair<double, double> gVal = lorentzianGradient(x, y, centerX, centerY, width);
                dxVals[i] = gVal.first;
                dyVals[i] = gVal.second;
                i++;
            }

    // Subtract the mean from the WF only (piston is not of interest).
    wf->subtractMean();

    return std::pair<spWF, spWFGrad>(wf, grad);
}

WFGLocalResponseGenerator::WFGLocalResponseGenerator(spPupil pupil)
    : mPupil(pupil)
{
}

double WFGLocalResponseGenerator::lorentzian(double x, double y, double centerX, double centerY, double width)
{
    double r2 = std::pow(x-centerX, 2) + std::pow(y-centerY, 2);
    return 1.0 / (width * r2 + 1);
}

std::pair<double, double> WFGLocalResponseGenerator::lorentzianGradient(double x, double y, double centerX, double centerY, double width)
{
    double r2 = std::pow(x-centerX, 2) + std::pow(y-centerY, 2);
    double denominator = std::pow(width * r2 + 1, 2);
    double numeratorX = - 2*width*(x-centerX);
    double numeratorY = - 2*width*(y-centerY);

    return std::pair<double, double>(numeratorX/denominator, numeratorY/denominator);
}
