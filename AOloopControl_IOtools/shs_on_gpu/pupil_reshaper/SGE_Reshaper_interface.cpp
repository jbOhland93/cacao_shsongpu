#include "SGE_Reshaper_interface.h"
#include "SGE_Reshaper.hpp"

extern "C"
{
    SGEReshapeHandle create_SGE_Reshaper(
        IMAGE* input,
        IMAGE* mask,
        int64_t linesAsSlices)
    {
        return new SGE_Reshaper(
            input,
            mask,
            linesAsSlices > 0);
    }

    void free_SGE_Reshaper(SGEReshapeHandle p)
    {
        delete (SGE_Reshaper*) p;
    }

    errno_t SGEE_reshape_do(SGEReshapeHandle p)
    {
        return ((SGE_Reshaper*) p)->reshapeDo();
    }
}
