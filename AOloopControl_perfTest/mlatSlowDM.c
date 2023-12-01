/**
 * @file    mlatSlowDM.c
 * @brief   measure hardware latency with slow DMs
 *
 * Measure latency between slow DM and WFS
 * "Slow" refers to an actuator rise time multiple times
 * the WFS frame duration, potentially including oscillations.
 * In this case, the change derivative, as measured by the regular
 * mlat, can be too noisy to effectively determine the latency.
 *
 * 
 */

#include "mlatSlowRecorder/MLS_Recorder_interface.h"

#include "CommandLineInterface/CLIcore.h"
#include <math.h>

static int cmdindex;

// Local variables pointers
static char *dmstream;
long         fpi_dmstream;

static char *wfsstream;
long         fpi_wfsstream;

static float *fpsMeasTime;
long          fpi_fpsMeasTime;

static uint32_t *pokePattern;
long             fpi_pokePattern;

static float *maxActStroke;
long          fpi_maxActStroke;

static uint32_t *numPokes;
long             fpi_numPokes;

static uint32_t *framesPerPoke;
long             fpi_framesPerPoke;

static int64_t *saveraw;
long            fpi_saveraw;

// Outputs
static float *framerateHz;
long          fpi_framerateHz;

static float *delayfr;
long          fpi_delayfr;

static float *delayus;
long          fpi_delayus;

static float *risetimefr;
long          fpi_risetimefr;

static float *risetimeus;
long          fpi_risetimeus;

static float *latencyfr;
long          fpi_latencyfr;

static float *latencyus;
long          fpi_latencyus;

static float *latencyrawfr;
long          fpi_latencyrawfr;

static float *latencyrawus;
long          fpi_latencyrawus;


static CLICMDARGDEF farg[] = {{
        CLIARG_STREAM,
        ".dmstream",
        "DM stream",
        "null",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &dmstream,
        &fpi_dmstream
    },
    {
        CLIARG_STREAM,
        ".wfsstream",
        "WFS stream",
        "null",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &wfsstream,
        &fpi_wfsstream
    },
    {
        CLIARG_FLOAT32,
        ".framerateMeasTime",
        "Timeframe to measure WFS framerate",
        "5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &fpsMeasTime,
        &fpi_fpsMeasTime
    },
    {
        CLIARG_UINT32,
        ".pokePattern",
        "0=homogene,1=sine,2=checkerboard",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &pokePattern,
        &fpi_pokePattern
    },
    {
        CLIARG_FLOAT32,
        ".maxActStroke",
        "Maximum actuator stroke in pattern",
        "0.1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &maxActStroke,
        &fpi_maxActStroke
    },
    {
        CLIARG_UINT32,
        ".numPokes",
        "Number of iterations",
        "100",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &numPokes,
        &fpi_numPokes
    },
    {
        CLIARG_UINT32,
        ".framesPerPoke",
        "Number frames in measurement sequence",
        "50",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &framesPerPoke,
        &fpi_framesPerPoke
    },
    {
        CLIARG_ONOFF,
        ".option.saveraw",
        "Save raw image cubes",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &saveraw,
        &fpi_saveraw
    },
    {
        CLIARG_FLOAT32,
        ".out.framerateHz",
        "WFS frame rate [Hz]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &framerateHz,
        &fpi_framerateHz
    },
    {
        CLIARG_FLOAT32,
        ".out.latencyRaw_fr",
        "unsmoothed hardware latency [frame]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &latencyrawfr,
        &fpi_latencyrawfr
    },
    {
        CLIARG_FLOAT32,
        ".out.latencyRaw_us",
        "unsmoothed hardware latency [us]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &latencyrawus,
        &fpi_latencyrawus
    },
    {
        CLIARG_FLOAT32,
        ".out.latency_fr",
        "smoothed hardware latency [frame]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &latencyfr,
        &fpi_latencyfr
    },
    {
        CLIARG_FLOAT32,
        ".out.latency_us",
        "smoothed hardware latency [us]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &latencyus,
        &fpi_latencyus
    },
    {
        CLIARG_FLOAT32,
        ".out.delay_fr",
        "delay of first movement detection [frame]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &delayfr,
        &fpi_delayfr
    },
    {
        CLIARG_FLOAT32,
        ".out.delay_us",
        "delay of first movement detection [us]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &delayus,
        &fpi_delayus
    },
    {
        CLIARG_FLOAT32,
        ".out.risetime_fr",
        "10\045 to [90\045|110\045] rise time [frame]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &risetimefr,
        &fpi_risetimefr
    },
    {
        CLIARG_FLOAT32,
        ".out.risetime_us",
        "10\045 to [90\045|110\045] rise time [us]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &risetimeus,
        &fpi_risetimeus
    },
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
        data.fpsptr->parray[fpi_dmstream].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;
        data.fpsptr->parray[fpi_wfsstream].fpflag |=
            FPFLAG_STREAM_RUN_REQUIRED | FPFLAG_CHECKSTREAM;

        data.fpsptr->parray[fpi_saveraw].fpflag |= FPFLAG_WRITERUN;
    }

    return RETURN_SUCCESS;
}

// Optional custom configuration checks.
// Runs at every configuration check loop iteration
//
static errno_t customCONFcheck()
{

    if(data.fpsptr != NULL)
    {}

    return RETURN_SUCCESS;
}

static CLICMDDATA CLIcmddata =
{
    "mlatslowdm", "measure latency between slow DM and WFS", CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Measure latency between two streams\n");

    printf(
        "Convention\n"
        "Latency is defined here as the time offset between:\n"
        "- Time at which input stream is perturbed\n"
        "- Average time of the last frame detecting less than\n"
        "  90%% or more than 110%% of the max stroke\n"
        "\n"
        "This is different to the regular mlat measurement as\n"
        "the scenario here is vastly different: unstead of a\n"
        "delayed but quasi instantaneous response of the DM,\n"
        "the DM is expected to feature a resolved rising slope.\n"
        ""
    );
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to DM
    IMGID imgdm = mkIMGID_from_name(dmstream);
    resolveIMGID(&imgdm, ERRMODE_ABORT);
    printf("DM size : %u %u\n", imgdm.md->size[0], imgdm.md->size[1]);

    // connect to WFS
    IMGID imgwfs = mkIMGID_from_name(wfsstream);
    resolveIMGID(&imgwfs, ERRMODE_ABORT);
    printf("WFS size : %u %u\n", imgwfs.md->size[0], imgwfs.md->size[1]);

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
    
    
    // === SET UP LATENCY RECORDER HERE
    MLSRHandle recorder = create_MLS_Recorder(
        data.fpsptr,
        imgdm.im,
        imgwfs.im,
        *fpsMeasTime,
        *pokePattern,
        *maxActStroke,
        *numPokes,
        *framesPerPoke,
        *saveraw);

    // === RECORD LATENCY
    mlsRecordDo(recorder);

/*    // === PUBLISH RESULTS
    printf("Results:\n");
    *framerateHz = getFPS_Hz(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.framerateHz");
    printf("\tFramerate = %.3f Hz\n", *framerateHz);
    *latencyfr = getHWlatency_frames(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.latency_fr");
    *latencyus = getHWlatency_us(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.latency_us");
    printf("\tHardware latency = %.3f us (%.3f frames)\n", *latencyus, *latencyfr);
    *delayfr = getHWdelay_frames(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.delay_fr");
    *delayus = getHWdelay_us(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.delay_us");
    printf("\tLatency to first motion = %.3f us (%.3f frames)\n", *delayus, *delayfr);
    *risetimefr = getRiseTime0to90_frames(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.risetime_fr");
    *risetimeus = getRiseTime0to90_us(recorder);
    functionparameter_SaveParam2disk(data.fpsptr, ".out.risetime_us");
    printf("\tRise time (10%% to [90%%|110%%]) = %.3f us (%.3f frames)\n", *risetimeus, *risetimefr);
*/
   
    // === FREE RECORDER
    free_MLS_Recorder(recorder);
    recorder = NULL;
    // === DONE

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

// Register function in CLI
errno_t
CLIADDCMD_AOloopControl_perfTest__mlat_slow_dm()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
