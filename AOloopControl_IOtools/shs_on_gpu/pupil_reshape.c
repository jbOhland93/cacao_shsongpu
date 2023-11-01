/**
 * @file    pupil_reshape.c
 * @brief   rearrange 1D data into a 2D pupil
 *
 */

#include "CommandLineInterface/CLIcore.h"
#include "pupil_reshaper/SGE_Reshaper_interface.h"
#include <math.h>

static int cmdindex;

// required for create_2Dimage_ID()
//#include "COREMOD_memory/COREMOD_memory.h"

// required for timespec_diff()
//#include "COREMOD_tools/COREMOD_tools.h"

// required for timespec_diff
//#include "CommandLineInterface/timeutils.h"

// Local variables pointers
// stream name of the input stream
static char *inputname;
// stream name of the mask stream
static char *maskname;

static int64_t *linesAsSlices;
static long      fpi_linesAsSlices = -1;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".input_name",
        "input image, 1D data in lines",
        "input", // default
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inputname,
        NULL
    },
    {
        CLIARG_IMG,
        ".mask_name",
        "stream with the pupil maks",
        "mask", // default
        CLIARG_VISIBLE_DEFAULT,
        (void **) &maskname,
        NULL
    },
    {
        CLIARG_ONOFF,
        ".linesAsSlices",
        "reshape input lines to 2D slices",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &linesAsSlices,
        &fpi_linesAsSlices
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
    "pupilReshape",
    "sort 1D data into a pupil",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}



static errno_t streamprocess(SGEReshapeHandle reshaper)
{
    DEBUG_TRACE_FSTART();
    
    // Code
    errno_t retVal = SGEE_reshape_do(reshaper);

    DEBUG_TRACE_FEXIT();
    return retVal;
}



static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inputimg = mkIMGID_from_name(inputname);
    resolveIMGID(&inputimg, ERRMODE_ABORT);
    IMGID maskimg = mkIMGID_from_name(maskname);
    resolveIMGID(&maskimg, ERRMODE_ABORT);

    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT

    // custom initialization
    printf(" COMPUTE Flags = %ld\n", CLIcmddata.cmdsettings->flags);
    if(CLIcmddata.cmdsettings->flags & CLICMDFLAG_PROCINFO)
    {
        // procinfo is accessible here
    }

    // === SET UP RESHAPER HERE
    printf("== Constructing reshaper ...\n");
    // Construct the reshaper
    SGEReshapeHandle reshaper = create_SGE_Reshaper(
        inputimg.im,
        maskimg.im,
        linesAsSlices);
    printf("== Reshaper constructed. Ready for reshaping.\n");
    // ===
    
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        streamprocess(reshaper);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    // === Cleaning up
    //processinfo_update_output_stream(processinfo, outimg.ID);
    printf("== Deleting reshaper.\n");
    free_SGE_Reshaper(reshaper);
    reshaper = NULL;
    // ===

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_AOloopControl_IOtools__pupilReshape()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
