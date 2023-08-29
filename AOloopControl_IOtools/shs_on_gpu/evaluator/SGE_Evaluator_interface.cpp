#include "SGE_Evaluator_interface.h"
#include "SGE_Evaluator.hpp"

extern "C"
{
    SGEEHandle create_SGE_Evaluator(
        IMAGE* ref,
        IMAGE* shscam,
        IMAGE* shsdark,
        const char* streamPrefix)
    {
        return new SGE_Evaluator(
            ref,
            shscam,
            shsdark,
            streamPrefix);
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
