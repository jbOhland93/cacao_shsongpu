#include "PupilZernikeGenerator_interface.h"
#include "PupilZernikeGenerator.hpp"

extern "C"
{
    PupilZernGenHandle create_PupilZernGenerator(
        IMAGE* zerncoeff,
        IMAGE* mask)
    {
        return new PupilZernikeGenerator(
            zerncoeff,
            mask);
    }

    void free_PupilZernGen(PupilZernGenHandle p)
    {
        delete (PupilZernikeGenerator*) p;
    }

    errno_t PupilZernGen_expand_do(PupilZernGenHandle p, int64_t output2D)
    {
        return ((PupilZernikeGenerator*) p)->expandDo(output2D > 0);
    }
}
