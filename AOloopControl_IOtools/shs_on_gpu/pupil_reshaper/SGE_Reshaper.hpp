#ifndef SGE_RESHAPER_HPP
#define SGE_RESHAPER_HPP

#include "../util/Pupil.hpp"
#include "../util/ImageHandler2D.hpp"
#include <errno.h>

// A class for rearranging 1D samples into a pupil of valid fields
class SGE_Reshaper
{
public:
    // Ctor, doing the initialization
    SGE_Reshaper(
        IMAGE* input,               // Input image stream
        IMAGE* mask,                // Stream holding the mask
        bool linesAsSlices = false);// Reshape lines into slices instead of vertical stacking

    // Triggers the reshaping
    errno_t reshapeDo();
private:
    // Input stream
    spImHandler2D(float) mp_IHinput;
    // The pupil representing valid and invalid fields
    spPupil mp_pupil;
    // Output stream
    spImHandler2D(float) mp_IHoutput;
};

#endif // SGE_RESHAPER_HPP
