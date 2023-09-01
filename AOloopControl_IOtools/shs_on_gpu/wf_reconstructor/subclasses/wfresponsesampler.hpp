#ifndef WFRESPONSESAMPLER_HPP
#define WFRESPONSESAMPLER_HPP

#include <vector>
#include "wfglocalresponsegenerator.hpp"

#define spRspnsSampler std::shared_ptr<ResponseSampler>
// A class to generate responses covering the full pupil plus a given proximity
class ResponseSampler {
public:
    static spRspnsSampler makeSampler(spPupil pupil, double proximity);

    std::vector<std::pair<spWF, spWFGrad>> generateSamples(double responseWidth);

private:
    spPupil mPupil;
    spWGLRspnsGen mResponseGenerator;
    double mMinPupilProximity;

    ResponseSampler(); // No publicly available constructor
    ResponseSampler(spPupil pupil, double proximity); // Private constructor
};

#endif // WFRESPONSESAMPLER_HPP
