#include "wfresponsesampler.hpp"
#include <cmath>
#include <iostream>

spRspnsSampler ResponseSampler::makeSampler(spPupil pupil, double proximity)
{
    return spRspnsSampler(new ResponseSampler(pupil, proximity));
}

std::vector<std::pair<spWF, spWFGrad>> ResponseSampler::generateSamples(double responseWidth)
{
    std::vector<std::pair<spWF, spWFGrad>> samples;

    int cProx = (int) ceil(mMinPupilProximity);

    for (int y = -cProx; y < mPupil->getHeight() + cProx; y++) {
        for (int x = -cProx; x < mPupil->getWidth() + cProx; x++) {
            if (mPupil->isInProximity(x, y, mMinPupilProximity)) {
                auto response = mResponseGenerator->generateResponse(x, y, responseWidth);
                samples.push_back(response);
            }
        }
    }

    return samples;
}

ResponseSampler::ResponseSampler(spPupil pupil, double proximity) 
    : mPupil(pupil), mMinPupilProximity(proximity)
{
    mResponseGenerator = WFGLocalResponseGenerator::makeLocalResponseGenerator(pupil);
}
