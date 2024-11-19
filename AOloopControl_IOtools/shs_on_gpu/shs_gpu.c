/**
 * @file    shs_gpu.c
 * @brief   shs evaluation on a gpu
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "evaluator/SGE_Evaluator_interface.h"
#include <math.h>

static int cmdindex;

// required for create_2Dimage_ID()
//#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
//#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
//#include "CommandLineInterface/timeutils.h"

// Local variables pointers
// stream name of the SHS camera
static char *camname;
// stream name of the SHS dark frame
static char *darkname;

// stream name of the SHS reference positions
static char *refPosName;
static char *refMaskName;
static char *refIntName;

// field to activate/deactivate the evaluation
static int64_t *evaluationOn;
static long     fpi_evaluationOn = -1;

// field to determine if the absolute or relative reference shall be used
static int64_t *absRef;
static long     fpi_absRef = -1;

// field to determine if the wf tilt shall be subtracted
static int64_t *removeTilt;
static long     fpi_removeTilt = -1;

// Toggle: calculate the WF from the gradient field
static int64_t *calcWF;
static long     fpi_calcWF = -1;

// Toggle: copy the evaluated gradient to the CPU
static int64_t *cpyGradToCPU;
static long     fpi_cpyGradToCPU = -1;

// Toggle: copy the WF to the CPU, if reconstructed
static int64_t *cpyWfToCPU;
static long     fpi_cpyWfToCPU = -1;

// Toggle: copy the intensity to the CPU
static int64_t *cpyIntToCPU;
static long     fpi_cpyIntToCPU = -1;

// Toggle: log wf PvT and RMS
static int64_t *logWfStats;
static long     fpi_logWfStats = -1;

static uint32_t *loopnumber;
static long      fpi_loopnumber = -1;

static char *loopname;
static long      fpi_loopname = -1;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".shscam",
        "shs camera image",
        "shscam",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &camname,
        NULL
    },
    {
        CLIARG_IMG,
        ".shsdark",
        "shs camera dark image",
        "shsdark",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &darkname,
        NULL
    },
    {
        CLIARG_IMG,
        ".reference.position",
        "stream holding the reference spot positions",
        "pos",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &refPosName,
        NULL
    },
    {
        CLIARG_IMG,
        ".reference.mask",
        "stream holding the reference mask",
        "mask",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &refMaskName,
        NULL
    },
    {
        CLIARG_IMG,
        ".reference.intensity",
        "stream holding the reference intensity",
        "int",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &refIntName,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".on_off",
        "toggle evaluation",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &evaluationOn,
        &fpi_evaluationOn
    },
    {
        CLIARG_ONOFF,
        ".comp.absRef",
        "toggle use of relative/absolute reference",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &absRef,
        &fpi_absRef
    },
    {
        CLIARG_ONOFF,
        ".comp.removeTilt",
        "toggle tilt subtraction",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &removeTilt,
        &fpi_removeTilt
    },
    {
        CLIARG_ONOFF,
        ".comp.calcWF",
        "toggle WF reconstruction",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &calcWF,
        &fpi_calcWF
    },
    {
        CLIARG_ONOFF,
        ".comp.cpyGradToCPU",
        "toggle copying the gradient to host memory",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cpyGradToCPU,
        &fpi_cpyGradToCPU
    },
    {
        CLIARG_ONOFF,
        ".comp.cpyWfToCPU",
        "toggle copying the WF to host memory",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cpyWfToCPU,
        &fpi_cpyWfToCPU
    },
    {
        CLIARG_ONOFF,
        ".comp.cpyIntToCPU",
        "toggle copying the intensity to host memory",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &cpyIntToCPU,
        &fpi_cpyIntToCPU
    },
    {
        CLIARG_ONOFF,
        ".comp.logWfStats",
        "log PtV and RMS of WF to file",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &logWfStats,
        &fpi_logWfStats
    },
    {
        CLIARG_UINT32,
        ".loopnumber",
        "The number of the AO loop, used for stream naming",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &loopnumber,
        &fpi_loopnumber
    },
    {
        CLIARG_STR,
        ".loopname",
        "The name of the AO loop, used for stream naming",
        "",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &loopname,
        &fpi_loopname
    }
};

// Optional custom configuration setup
// Runs once at conf startup
//
// To use this function, set :
// CLIcmddata.FPS_customCONFsetup = customCONFsetup
// when registering function
// (see end of this file)
//
static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
        // can toggle while running
        data.fpsptr->parray[fpi_evaluationOn].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_absRef].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_removeTilt].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_calcWF].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_cpyGradToCPU].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_cpyWfToCPU].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_cpyIntToCPU].fpflag |= FPFLAG_WRITERUN;
        data.fpsptr->parray[fpi_logWfStats].fpflag |= FPFLAG_WRITERUN;
    }

    // increment counter at every configuration check
    /* Skip tests for now
    *cntindex = *cntindex + 1;

    if(*cntindex >= *cntindexmax)
    {
        *cntindex = 0;
    }
    */

    return RETURN_SUCCESS;
}

// Optional custom configuration checks
// Runs at every configuration check loop iteration
//
// To use this function, set :
// CLIcmddata.FPS_customCONFcheck = customCONFcheck
// when registering function
// (see end of this file)
//
static errno_t customCONFcheck()
{
    /* Skip tests for now
    if(data.fpsptr != NULL)
    {
        if(data.fpsptr->parray[fpi_ex0mode].fpflag & FPFLAG_ONOFF)  // ON state
        {
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag |= FPFLAG_VISIBLE;
        }
        else // OFF state
        {
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_USED;
            data.fpsptr->parray[fpi_ex1mode].fpflag &= ~FPFLAG_VISIBLE;
        }

        // increment counter at every configuration check
        *cntindex = *cntindex + 1;

        if(*cntindex >= *cntindexmax)
        {
            *cntindex = 0;
        }

    }
    */

    return RETURN_SUCCESS;
}



static CLICMDDATA CLIcmddata =
{
    "shsGpuEval",
    "evaluate a SHS image on a GPU",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t streamprocess(SGEEHandle evaluator)
{
    DEBUG_TRACE_FSTART();
    
    // Code
    errno_t retVal = SGEE_eval_do(
        evaluator,
        *absRef,
        *removeTilt,
        *calcWF,
        *cpyGradToCPU,
        *cpyWfToCPU,
        *cpyIntToCPU,
        *logWfStats);

    DEBUG_TRACE_FEXIT();
    return retVal;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID camimg = mkIMGID_from_name(camname);
    resolveIMGID(&camimg, ERRMODE_ABORT);
    IMGID darkimg = mkIMGID_from_name(darkname);
    resolveIMGID(&darkimg, ERRMODE_ABORT);
    IMGID posimg = mkIMGID_from_name(refPosName);
    resolveIMGID(&posimg, ERRMODE_ABORT);
    IMGID maskimg = mkIMGID_from_name(refMaskName);
    resolveIMGID(&maskimg, ERRMODE_ABORT);
    IMGID intimg = mkIMGID_from_name(refIntName);
    resolveIMGID(&intimg, ERRMODE_ABORT);

    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    // === SET UP EVALUATOR HERE
    printf("== Constructing evaluator ...\n");
    // Allocate a buffer for the stream prefix
    const char* funPrefix = "_shsEval_";
    uint8_t fpLen = strlen(funPrefix);
    char loopPrefix[(int)((
        3                           // "aol"
        +ceil(log10(*loopnumber))   // loopnumber
        +1                          // "_"
        // +lnLen                      // loopname
        +fpLen                      // function prefix
        )*sizeof(char))];
    // Build the stream prefix
    sprintf(loopPrefix, "%s%d%s",
        "aol",
        *loopnumber,
        //"_",
        //loopname,
        funPrefix);
    // Construct the evaluator
    SGEEHandle evaluator = create_SGE_Evaluator(
        data.fpsptr,
        camimg.im,
        darkimg.im,
        posimg.im,
        maskimg.im,
        intimg.im,
        loopPrefix);
    printf("== Evaluator constructed. Ready for evaluation.\n");
    // ===
    
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        if (*evaluationOn)
            streamprocess(evaluator);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    // === EVALUATING RESULTS HERE
    //processinfo_update_output_stream(processinfo, outimg.ID);
    printf("== Deleting evaluator.\n");
    free_SGE_Evaluator(evaluator);
    evaluator = NULL;
    // ===

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_AOloopControl_IOtools__EvaluateShsGPU()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
