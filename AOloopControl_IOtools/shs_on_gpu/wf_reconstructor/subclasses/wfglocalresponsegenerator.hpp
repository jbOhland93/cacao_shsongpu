#ifndef WFGLOCALRESPONSEGENERATOR_HPP
#define WFGLOCALRESPONSEGENERATOR_HPP

#include "wavefront.hpp"
#include "wfgrad.hpp"

#define spWGLRspnsGen std::shared_ptr<WFGLocalResponseGenerator>

// A class that can generate spatial response functions on a pupil in WF and Gradient space
class WFGLocalResponseGenerator {
public:
    static spWGLRspnsGen makeLocalResponseGenerator(spPupil pupil);

    std::pair<spWF, spWFGrad> generateResponse(double centerX, double centerY, double width);

private:
    spPupil mPupil;

    WFGLocalResponseGenerator(); // No publicly available constructor
    WFGLocalResponseGenerator(spPupil pupil); // Private constructor

    double lorentzian(double x, double y, double centerX, double centerY, double width);
    std::pair<double, double> lorentzianGradient(double x, double y, double centerX, double centerY, double width);
};

#endif // WFGLOCALRESPONSEGENERATOR_HPP