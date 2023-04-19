#include "SGE_Evaluator_interface.h"
#include "SGE_Evaluator.hpp"

extern "C"
{
    SGEEHandle create_SGE_Evaluator(
        IMAGE* in,
        IMAGE* dark)
    {
        return new SGE_Evaluator(
            in,
            dark);
    }

    void free_SGE_Evaluator(SGEEHandle p)
    {
        delete (SGE_Evaluator*) p;
    }

    errno_t SGEE_eval_do(SGEEHandle p)
    {
        return ((SGE_Evaluator*) p)->evaluateDo();
    }
}
