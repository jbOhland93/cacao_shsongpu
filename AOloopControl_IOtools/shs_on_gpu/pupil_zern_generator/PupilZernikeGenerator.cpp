#include "PupilZernikeGenerator.hpp"
#include "CLIcore.h"

PupilZernikeGenerator::PupilZernikeGenerator(IMAGE* zerncoeff, IMAGE* mask)
{
    // Adopt input stream
    if (!checkAtype<float>(zerncoeff->md->datatype))
        throw std::runtime_error("PupilZernikeGenerator: the coefficient stream has to be of type float.\n");
    mp_IHcoeffs = ImageHandler2D<float>::newHandler2DAdoptImage(zerncoeff->name);

    // Set up pupil
    if (!checkAtype<uint8_t>(mask->md->datatype))
        throw std::runtime_error("PupilZernikeGenerator: the mask stream has to be of type uint8_t.\n"); 
    mp_pupil = Pupil::makePupil(
        mask->md->size[0],
        mask->md->size[1],
        (uint8_t*) ImageStreamIO_get_image_d_ptr(mask));
    mp_circle = Circle::makeCircle(mp_pupil);
    mp_circle->print();
    

    // Set up output
    /*std::string outputName = "";
    outputName.append("_reshape");
    if (!linesAsSlices)
    {   // Order each reshaped pupil below the last one.
        mp_IHoutput = ImageHandler2D<float>::newImageHandler2D(
            outputName,
            mp_pupil->getWidth(),
            mp_pupil->getHeight()*numFrames);
    }
    else
    {   // Use each reshaped pupil as a slice
        mp_IHoutput = ImageHandler2D<float>::newImageHandler2D(
            outputName,
            mp_pupil->getWidth(),
            mp_pupil->getHeight(),
            numFrames);
    }
    mp_IHoutput->setPersistent(true);*/
}

errno_t PupilZernikeGenerator::expandDo(bool output2D)
{
    printf("EXPAND NOT DONE YET\n");
}