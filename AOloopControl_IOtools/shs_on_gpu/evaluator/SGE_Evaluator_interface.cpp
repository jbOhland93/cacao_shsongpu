#include "SGE_Evaluator_interface.h"
#include "SGE_Evaluator.hpp"

extern "C"
{
    SGEEHandle create_SGE_Evaluator(
        FUNCTION_PARAMETER_STRUCT* fps,
        IMAGE* shscam,
        IMAGE* shsdark,
        IMAGE* refPos,
        IMAGE* refMask,
        IMAGE* refInt,
        const char* streamPrefix)
    {
        return new SGE_Evaluator(
            fps,
            shscam,
            shsdark,
            refPos,
            refMask,
            refInt,
            streamPrefix);
    }

    void free_SGE_Evaluator(SGEEHandle p)
    {
        delete (SGE_Evaluator*) p;
    }

    errno_t SGEE_eval_do(
        SGEEHandle p,
        int64_t useAbsRef,
        int64_t removeTilt,
        int64_t calcWF,
        int64_t cpyGradToCPU,
        int64_t cpyWfToCPU,
        int64_t cpyIntToCPU,
        int64_t logWFstats)
    {
        return ((SGE_Evaluator*) p)->evaluateDo(
            useAbsRef != 0,
            removeTilt != 0,
            calcWF != 0,
            cpyGradToCPU != 0,
            cpyWfToCPU != 0,
            cpyIntToCPU != 0,
            logWFstats != 0);
    }
}
