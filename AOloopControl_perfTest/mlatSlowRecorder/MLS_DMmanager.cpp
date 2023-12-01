#include "MLS_DMmanager.hpp"

MLS_DMmanager::MLS_DMmanager(
        IMAGE* dmstream,        // Stream of the DM input
        PokePattern pokePattern,// Poke pattern
        float maxActStroke)     // Maximum actuator stroke in pattern
        :   m_pokePattern(pokePattern),
            m_maxActStroke(maxActStroke)
{
    mp_IHdm = ImageHandler2D<float>::newHandler2DAdoptImage(dmstream->name);

    // Make DM patterns
    std::string dmStreamName = mp_IHdm->getImage()->name;

    std::string poke0name = dmStreamName;
    poke0name += "_mlatPoke0";
    mp_IHdmPoke0 = ImageHandler2D<float>::newHandler2DfrmImage(
        poke0name,
        mp_IHdm->getImage());
    float* dptr0 = mp_IHdmPoke0->getWriteBuffer();

    std::string poke1name = dmStreamName;
    poke1name += "_mlatPoke1";
    mp_IHdmPoke1 = ImageHandler2D<float>::newHandler2DfrmImage(
        poke1name,
        mp_IHdm->getImage());
    float* dptr1 = mp_IHdmPoke1->getWriteBuffer();

    for (int iy = 0; iy < mp_IHdm->mHeight; iy++)
        for (int ix = 0; ix < mp_IHdm->mWidth; ix++)
        {
            int i = mp_IHdm->mWidth * iy + ix;
            dptr0[i] = 0;
            switch (m_pokePattern)
            {
            case PokePattern::HOMOGENEOUS:
                dptr1[i] = m_maxActStroke;
                break;
            case PokePattern::CHECKERBOARD:
                dptr1[i] = m_maxActStroke * (( ix + iy % 2 ) % 2);
                break;
            case PokePattern::SINE:
                dptr1[i] = m_maxActStroke
                            * cos(20 * ix/mp_IHdm->mWidth)
                            * cos(20 * iy/mp_IHdm->mWidth);
                break;
            default:
                throw std::runtime_error("Unknown PokePattern.");
            }
        }
    mp_IHdmPoke0->updateWrittenImage();
    mp_IHdmPoke1->updateWrittenImage();
}



void MLS_DMmanager::preloadDM(bool poke)
{
    if (poke)
        mp_IHdm->cpy(mp_IHdmPoke1->getImage(), false);
    else
        mp_IHdm->cpy(mp_IHdmPoke0->getImage(), false);
}



void MLS_DMmanager::triggerDM()
{
    mp_IHdm->updateWrittenImage();
}