#include "MLS_DMmanager.hpp"

MLS_DMmanager::MLS_DMmanager(
        IMAGE* dmstream,            // Stream of the DM input
        PokePattern pokePattern,    // Poke pattern
        std::string patternImage,   // Stream of DM patterns
        uint32_t shmPatternIdx,     // Index of shmIm pattern slice
        float maxActStroke)     // Maximum actuator stroke in pattern
        :   m_pokePattern(pokePattern),
            m_maxActStroke(maxActStroke)
{
    mp_IHdm = ImageHandler2D<float>::newHandler2DAdoptImage(dmstream->name);
    float shmImPatternNorm = 0;
    if (pokePattern == PokePattern::SHMIM)
    {
        printf("===== SHMIM FTW ====\n");
        mp_IHpatterns = ImageHandler2D<float>::newHandler2DAdoptImage(patternImage);
        mp_IHpatterns->setSlice(shmPatternIdx);
        float max = mp_IHpatterns->getMaxInROI();
        float min = mp_IHpatterns->getMinInROI();
        shmImPatternNorm = max > -min? max : -min;
        shmImPatternNorm = m_maxActStroke / shmImPatternNorm;
        printf("===== SHMIM NORM = %.9f ====\n", shmImPatternNorm);
    }

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

    float h = mp_IHdm->mHeight;
    float w = mp_IHdm->mWidth;
    for (int iy = 0; iy < mp_IHdm->mHeight; iy++)
        for (int ix = 0; ix < mp_IHdm->mWidth; ix++)
        {
            int i = mp_IHdm->mWidth * iy + ix;
            dptr0[i] = 0;
            switch (m_pokePattern)
            {
            case PokePattern::SHMIM:
                dptr1[i] = mp_IHpatterns->read(ix, iy) * shmImPatternNorm;
                break;
            case PokePattern::HOMOGENEOUS:
                dptr1[i] = m_maxActStroke;
                break;
            case PokePattern::CHECKERBOARD:
                dptr1[i] = m_maxActStroke * (2*(( ix + iy % 2 ) % 2)-1);
                break;
            case PokePattern::SINE:
                dptr1[i] = m_maxActStroke
                            * cos(20 * ix/w)
                            * cos(20 * iy/h);
                break;
            case PokePattern::SQUARE:
                dptr1[i] = m_maxActStroke
                            * (cos(20 * ix/w)
                            * cos(20 * iy/h) > 0 ? 1 : -1);
                break;
            case PokePattern::HALFSQUARE:
                dptr1[i] = m_maxActStroke
                            * (cos(10 * ix/w)
                            * cos(10 * iy/h) > 0 ? 1 : -1);
                break;
            case PokePattern::DOUBLESQUARE:
                dptr1[i] = m_maxActStroke
                            * (cos(40 * ix/w)
                            * cos(40 * iy/h) > 0 ? 1 : -1);
                break;
            case PokePattern::XRAMP:
                dptr1[i] = m_maxActStroke
                            * (2*ix/w - 1);
                break;
            case PokePattern::XHALF:
                dptr1[i] = m_maxActStroke
                            * (ix+1 > w/2 ? -1 : 1);
                break;
            case PokePattern::YRAMP:
                dptr1[i] = m_maxActStroke
                            * (2*iy/h - 1);
                break;
            case PokePattern::YHALF:
                dptr1[i] = m_maxActStroke
                            * (iy+1 > h/2 ? -1 : 1);
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