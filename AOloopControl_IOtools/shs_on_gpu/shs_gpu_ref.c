/**
 * @file    shs_gpu_ref.c
 * @brief   reference recording for shs evaluation on gpu
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "ref_recorder/SGR_Recorder_interface.h"
#include <math.h>

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

static float *campxsize;
static long      fpi_campxsize = -1;

static float *mlapitch;
static long      fpi_mlapitch = -1;

static float *shsfoclen;
static long      fpi_shsfoclen = -1;

static float *minRelIntensity;
static long      fpi_minRelIntensity = -1;

static float *minSpotPrec;
static long      fpi_minSpotPrec = -1;

static int64_t *visualize;
static long      fpi_visualize = -1;

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
    },
    {
        CLIARG_FLOAT32,
        ".shs.campixsize",
        "camera pixel size in um",
        "13.7",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &campxsize,
        &fpi_campxsize
    },
    {
        CLIARG_FLOAT32,
        ".shs.mlapitch",
        "MLA pitch in um",
        "250.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &mlapitch,
        &fpi_mlapitch
    },
    {
        CLIARG_FLOAT32,
        ".shs.shsfoclen",
        "MLS-to-sensor distance in um",
        "11330.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &shsfoclen,
        &fpi_shsfoclen
    },
    {
        CLIARG_FLOAT32,
        ".mask.minRelInt",
        "Minimum subaperture intensity comapred to max [0:1]",
        "0.2",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &minRelIntensity,
        &fpi_minRelIntensity
    },
    {
        CLIARG_FLOAT32,
        ".mask.minSpotPrec",
        "Minimum spot precision in urad, for generating a mask",
        "60.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &minSpotPrec,
        &fpi_minSpotPrec
    },
    {
        CLIARG_ONOFF,
        ".option.visualize",
        "Generates additional image streams to visually assess the process",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &visualize,
        &fpi_visualize
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
    "shsGpuRef",
    "record a SHS ref for GPU evaluation",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t streamprocess(SGRRHandle recorder)
{
    DEBUG_TRACE_FSTART();
    
    // Trigger recorder
    errno_t returnval = SGRR_sample_do(recorder);
    // Print the current state
    printf("SGR recorder status: %s", get_SGRR_state_descr(recorder));

    DEBUG_TRACE_FEXIT();
    return returnval;
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
    long loopcnt = 0;
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
        loopcnt = CLIcmddata.cmdsettings->procinfo_loopcntMax;
    }
    
    // === SET UP REF RECORDER HERE
    SGRRHandle recorder = create_SGR_Recorder(
        data.fpsptr,
        inimg.im,
        darkimg.im,
        *campxsize,
        *mlapitch,
        *shsfoclen,
        loopcnt,
        *visualize);
    printf("\nSGR recorder status: %s", get_SGRR_state_descr(recorder));
    // === RECORDER SETUP DONE

    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        streamprocess(recorder);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    // === EVALUATING RESULTS HERE
    errno_t err = SGRR_evaluate_rec_buffers(
        recorder,
        *minRelIntensity,
        *minSpotPrec);
    //processinfo_update_output_stream(processinfo, outimg.ID);
    
    if (err == RETURN_SUCCESS)
        printf("\nEvaluation done, exiting.\n");
    else
        printf("\nERROR: %s\n", get_SGRR_state_descr(recorder));
    
    free_SGR_Recorder(recorder);
    recorder = NULL;
    // === EVAULATION DONE

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_AOloopControl_IOtools__recordShsRefGPU()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
