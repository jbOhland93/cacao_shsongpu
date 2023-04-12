/**
 * @file    shs_gpu_ref.c
 * @brief   reference recording for shs evaluation on gpu
 *
 */

#include "CommandLineInterface/CLIcore.h"
//#include "ref_recorder/SGR_Recorder_interface.h"
//#include <math.h>

static int cmdindex;

// required for create_2Dimage_ID()
//#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
//#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
//#include "CommandLineInterface/timeutils.h"

// Local variables pointers

static char *inimname;

static char *darkimname;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in_name",
        "input image",
        "cam",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname,
        NULL
    },
    {
        CLIARG_IMG,
        ".dark",
        "darkframe",
        "dark",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &darkimname,
        NULL
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
    "evaluate an SHS image stream harnessing the unfathomable power of GPUs",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t streamprocess()
{
    DEBUG_TRACE_FSTART();
    
    // Code
    printf("SHS on GPU code is executed ...\n");

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg = mkIMGID_from_name(inimname);
    resolveIMGID(&inimg, ERRMODE_ABORT);
    IMGID darkimg = mkIMGID_from_name(darkimname);
    resolveIMGID(&darkimg, ERRMODE_ABORT);

    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }
    
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        streamprocess();
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    //processinfo_update_output_stream(processinfo, outimg.ID);

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
