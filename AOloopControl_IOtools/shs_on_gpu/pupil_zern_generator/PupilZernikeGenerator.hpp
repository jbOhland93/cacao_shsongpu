#ifndef PUPILZERNIKEGENERATOR_HPP
#define PUPILZERNIKEGENERATOR_HPP

#include "../util/Circle.hpp"
#include "../util/ImageHandler2D.hpp"
#include <errno.h>

// A class for expanding Zernike coefficients on a pupil of valid fields
class PupilZernikeGenerator
{
public:
    // Ctor, doing the initialization
    PupilZernikeGenerator(
        IMAGE* zerncoeff,           // Input Zernike coefficient stream
        IMAGE* mask);                // Stream holding the mask

    // Triggers the expansion
    errno_t expandDo(bool output2D);
private:
    // Coefficient stream
    spImHandler2D(float) mp_IHcoeffs;
    // The pupil representing valid and invalid fields
    spPupil mp_pupil;
    // The unit circle, used for the expansion
    spCircle mp_circle;
    // Output streams
    spImHandler2D(float) mp_IHoutput1D = nullptr;
    spImHandler2D(float) mp_IHoutput2D = nullptr;
};

#endif // PUPILZERNIKEGENERATOR_HPP
