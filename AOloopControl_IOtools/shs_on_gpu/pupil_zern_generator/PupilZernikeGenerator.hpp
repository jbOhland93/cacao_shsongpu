#ifndef PUPILZERNIKEGENERATOR_HPP
#define PUPILZERNIKEGENERATOR_HPP

#include "../util/Circle.hpp"
#include "../util/ImageHandler2D.hpp"
#include "../pupil_reshaper/SGE_Reshaper.hpp"
#include <errno.h>

// A class for expanding Zernike coefficients on a pupil of valid fields
class PupilZernikeGenerator
{
public:
    // Ctor, doing the initialization
    PupilZernikeGenerator(
        IMAGE* zerncoeff,           // Input Zernike coefficient stream
        IMAGE* mask,                // Stream holding the mask
        std::string outputName);    // Name of the destination image

    // Triggers the expansion
    errno_t expandDo(bool output2D);
private:
    // Mask stream
    spImHandler2D(uint8_t) mp_IHmask;
    // Coefficient stream
    spImHandler2D(float) mp_IHcoeffs;
    // The pupil representing valid and invalid fields
    spPupil mp_pupil;
    // The unit circle, used for the expansion
    spCircle mp_circle;
    // Coordinate grid on the pupil
    std::vector<Point<double>> m_polarGrid;
    // Reshaper for 2D output
    std::shared_ptr<SGE_Reshaper> mp_pupilReshaper = nullptr;
    // Output streams
    spImHandler2D(float) mp_IHoutput1D = nullptr;
    spImHandler2D(float) mp_IHoutput2D = nullptr;
};

#endif // PUPILZERNIKEGENERATOR_HPP
