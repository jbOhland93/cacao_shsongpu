#include "SGE_Reshaper.hpp"
#include "CLIcore.h"

SGE_Reshaper::SGE_Reshaper(
        IMAGE* input,               // Input image stream
        IMAGE* mask,                // Stream holding the mask
        bool linesAsSlices)         // Reshape lines into slices instead of vertical stacking
{
    // Adopt input stream
    if (!checkAtype<float>(input->md->datatype))
        throw std::runtime_error("SGE_Reshaper: the input stream has to be of type float.\n");
    mp_IHinput = ImageHandler2D<float>::newHandler2DAdoptImage(input->name);

    // Set up pupil
    if (!checkAtype<uint8_t>(mask->md->datatype))
        throw std::runtime_error("SGE_Reshaper: the mask stream has to be of type uint8_t.\n"); 
    mp_pupil = Pupil::makePupil(
        mask->md->size[0],
        mask->md->size[1],
        (uint8_t*) ImageStreamIO_get_image_d_ptr(mask));

    // Check sizes
    if (mp_pupil->getNumValidFields() != mp_IHinput->mWidth)
    {   
        printf("ERROR: Size mismatch! Expected width: %d. Actual width: %d.\n",
            mp_pupil->getNumValidFields(),
            mp_IHinput->mWidth);
        throw std::runtime_error(
            "SGE_Reshaper: the width of the input image does not correspond to the number of valid fields in the pupil.\n");
    }
    // Each line of each slice of the input corresponds to one reshaped pupil
    uint32_t numFrames = mp_IHinput->mHeight * mp_IHinput->mDepth;

    // Set up output
    // Each line of the input is one frame
    
    std::string outputName = input->name;
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
    mp_IHoutput->setPersistent(true);
}

errno_t SGE_Reshaper::reshapeDo()
{
    float* readBuf = mp_IHinput->getWriteBuffer();
    float* writeBuf = mp_IHoutput->getWriteBuffer();

    // Each line of each slice of the input corresponds to one reshaped pupil
    uint32_t numFrames = mp_IHinput->mHeight * mp_IHinput->mDepth;
    for (uint32_t i = 0; i < numFrames; i++)
    {   // For every frame
        mp_pupil->fill2DarrWithValues(
            mp_IHinput->mWidth,
            readBuf,
            mp_pupil->get2DarraySize(),
            writeBuf,
            NAN);
        // Go to next frame
        readBuf += mp_IHinput->mWidth;
        writeBuf += mp_pupil->get2DarraySize();
    }
    mp_IHoutput->updateWrittenImage();

    return RETURN_SUCCESS;
}
