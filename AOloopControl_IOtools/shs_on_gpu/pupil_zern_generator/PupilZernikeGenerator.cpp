#include "PupilZernikeGenerator.hpp"
#include "CLIcore.h"
extern "C" {
#include "../../../../milk-extra-src/ZernikePolyn/zernike_value.h"
}

PupilZernikeGenerator::PupilZernikeGenerator(
    IMAGE* zerncoeff,
    IMAGE* mask,
    std::string outputName)
{
    // Adopt input stream
    if (!checkAtype<float>(zerncoeff->md->datatype))
        throw std::runtime_error("PupilZernikeGenerator: the coefficient stream has to be of type float.\n");
    mp_IHcoeffs = ImageHandler2D<float>::newHandler2DAdoptImage(zerncoeff->name);

    // Set up pupil
    if (!checkAtype<uint8_t>(mask->md->datatype))
        throw std::runtime_error("PupilZernikeGenerator: the mask stream has to be of type uint8_t.\n"); 
    mp_IHmask = ImageHandler2D<uint8_t>::newHandler2DAdoptImage(mask->name);
    mp_pupil = Pupil::makePupil(
        mp_IHmask->mWidth,
        mp_IHmask->mHeight,
        mp_IHmask->getWriteBuffer());
    mp_circle = Circle::makeCircle(mp_pupil);
    mp_circle->print();
    
    // Span polar grid over the pupil
    // This is used as input for the zernike polynomes.
    int pupilDatLen;
    bool* pupilDat = mp_pupil->getDataPtr(&pupilDatLen);
    for (int ix = 0; ix < mp_pupil->getWidth(); ix++)
        for (int iy = 0; iy < mp_pupil->getHeight(); iy++)
            if (pupilDat[iy*mp_pupil->getWidth() + ix])
                m_polarGrid.push_back(mp_circle->toPolar(Point<double>(ix, iy)));
    
    // Set up output
    try
    {
        mp_IHoutput1D = ImageHandler2D<float>::newHandler2DAdoptImage(outputName);
        if (mp_IHoutput1D->mWidth != mp_pupil->getNumValidFields()
            || mp_IHoutput1D->mHeight != 1)
        {
            std::string err("Output image exists, but has wrong size. ");
            err.append("Overwriting image.");
            mp_IHoutput1D->setPersistent(false);
            mp_IHoutput1D = nullptr;
            throw std::runtime_error(err.c_str());
        }
    }
    catch(const std::runtime_error& e)
    {
        printf("Error opening image:\n%s\n", e.what());
        mp_IHoutput1D = ImageHandler2D<float>::newImageHandler2D(
            outputName,
            mp_pupil->getNumValidFields(),
            1);
        mp_IHoutput1D->setPersistent(true);
    }

    // Initialize zernike base
    zernike_init();
}

errno_t PupilZernikeGenerator::expandDo(bool output2D)
{
    for (int i = 0; i < m_polarGrid.size(); i++)
    {
        double val = 0;
        for (long z = 0; z < mp_IHcoeffs->mWidth; z++)
            val += mp_IHcoeffs->read(z,0) *
                    Zernike_value(z, m_polarGrid[i].mX, m_polarGrid[i].mY);
        mp_IHoutput1D->write((float) val, i, 0);
    }
    mp_IHoutput1D->updateWrittenImage();

    // If 2D output is required, reshape the current output
    if (!output2D)
        return EXIT_SUCCESS;
    else
    {
        if (mp_pupilReshaper == nullptr)
            mp_pupilReshaper = std::make_shared<SGE_Reshaper>(
                mp_IHoutput1D->getImage(), mp_IHmask->getImage());

        return mp_pupilReshaper->reshapeDo();
    }
}