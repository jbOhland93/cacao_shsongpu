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
// stream name of the SHS reference positions
static char *refname;
// stream name of the SHS camera
static char *camname;
// stream name of the SHS dark frame
static char *darkname;

// field to activate/deactivate the evaluation
static int64_t *evaluationOn;
static long     fpi_evaluationOn = -1;

// field to determine if the absolute or relative reference shall be used
static int64_t *absRef;
static long     fpi_absRef = -1;

static uint32_t *loopnumber;
static long      fpi_loopnumber = -1;

static char *loopname;
static long      fpi_loopname = -1;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".ref_name",
        "reference image",
        "cam",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &refname,
        NULL
    },
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
        ".absRef",
        "toggle use of relative/absolute reference",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &absRef,
        &fpi_absRef
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



static errno_t streamprocess(SGEEHandle evaluator, int64_t useAbsoluteReference)
{
    DEBUG_TRACE_FSTART();
    
    // Code
    errno_t retVal = SGEE_eval_do(evaluator, useAbsoluteReference);

    DEBUG_TRACE_FEXIT();
    return retVal;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID refimg = mkIMGID_from_name(refname);
    resolveIMGID(&refimg, ERRMODE_ABORT);
    IMGID camimg = mkIMGID_from_name(camname);
    resolveIMGID(&camimg, ERRMODE_ABORT);
    IMGID darkimg = mkIMGID_from_name(darkname);
    resolveIMGID(&darkimg, ERRMODE_ABORT);

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
        //+1                          // "_"
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
        refimg.im,
        camimg.im,
        darkimg.im,
        loopPrefix);
    printf("== Evaluator constructed. Ready for evaluation.\n");
    // ===
    
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        if (*evaluationOn)
            streamprocess(evaluator, *absRef);
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
