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

static int64_t *skipMFramerate;
long            fpi_skipMFramerate;

static float *fpsMeasTime;
long          fpi_fpsMeasTime;

static uint32_t *numPokes;
long             fpi_numPokes;

static uint32_t *framesPerPoke;
long             fpi_framesPerPoke;

static int64_t *saveraw;
long            fpi_saveraw;

// Poke pattern settings
static int32_t *pokePatternType;
long             fpi_pokePatternType;

static char *customPatternStream;
long         fpi_customPatternStream;

static uint32_t *customPatternSliceIdx;
long             fpi_customPatternSliceIdx;

static float *patternToStrokeMul;
long          fpi_patternToStrokeMul;

static int64_t *useCustomResponseStream;
long            fpi_useCustomResponseStream;

static char *customResponseStream;
long         fpi_customResponseStream;

static uint32_t *customResponseSliceIdx;
long             fpi_customResponseSliceIdx;

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
        CLIARG_ONOFF,
        ".skipFramerateMeas",
        "Skip the framerate measurement prior to latency",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &skipMFramerate,
        &fpi_skipMFramerate
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
        CLIARG_INT32,
        ".pattern.type",
        "-1=custom,0=homogene,1=sine,... - see help",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &pokePatternType,
        &fpi_pokePatternType
    },
    {
        CLIARG_STR,
        ".pattern.customPokeStream",
        "Stream where slices are pokable DM channel values",
        "null",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &customPatternStream,
        &fpi_customPatternStream
    },
    {
        CLIARG_UINT32 ,
        ".pattern.customPatternIdx",
        "slice index of the pattern stream to poke",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &customPatternSliceIdx,
        &fpi_customPatternSliceIdx
    },
    {
        CLIARG_FLOAT32,
        ".pattern.pokeAmpMul",
        "pattern-to-stroke factor",
        "0.1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &patternToStrokeMul,
        &fpi_patternToStrokeMul
    },
    {
        CLIARG_ONOFF,
        ".pattern.useCustomResponse",
        "if ON: don't record the response but use the custom one",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &useCustomResponseStream,
        &fpi_useCustomResponseStream
    },
    {
        CLIARG_STR,
        ".pattern.customResponseStream",
        "Stream where slices are WFS modes",
        "null",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &customResponseStream,
        &fpi_customResponseStream
    },
    {
        CLIARG_UINT32 ,
        ".pattern.customResponseIdx",
        "slice index of the response stream",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &customResponseSliceIdx,
        &fpi_customResponseSliceIdx
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
        "\n"
        "The pattern which will be used on the DM can be set via\n"
        "the .pattern.type parameter. Currently, there are:\n"
        "-> -1=custom - SHM via .pattern.customPokeStream & .pattern.customPatternIdx\n"
        "->  0=homogeneous - all actuators to 1\n"
        "->  1=sine - sine grid with ~6 oscillations over the aperture\n"
        "->  2=checkerboard - neighbouring actuators alternate to +/- 1\n"
        "->  3=square - like sine, but 1 if positive and -1 if negative\n"
        "->  4=large square - double the size as square\n"
        "->  5=small square - half the size as square\n"
        "->  6=x-ramp - tilt in x-direction from -1 to 1\n"
        "->  7=x-half - left half is -1, right half is 1\n"
        "->  8=y-ramp - tilt in y-direction from -1 to 1\n"
        "->  9=y-half - upper half is -1, lower half is 1\n"
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

    // connect to pattern stream if required, and check dimensions
    IMGID imPattern;
    if (*pokePatternType == -1)
    {
        printf("READ RESULT: %ld\n", read_sharedmem_image(customPatternStream));
        imPattern = mkIMGID_from_name(customPatternStream);
        resolveIMGID(&imPattern, ERRMODE_ABORT);
        if (imPattern.md->naxis != imgdm.md->naxis + 1)
        {
            printf("Pattern stream has wrong dimensinality. Has to be DM stream dim +1.\n");
            return RETURN_FAILURE;
        }
        for (int i = 0; i < imgdm.md->naxis; i++)
            if (imPattern.md->size[i] != imgdm.md->size[i])
            {
                printf("Pattern stream has wrong pattern size. Must match DM stream size.\n");
                return RETURN_FAILURE;
            }
        if (imPattern.md->size[imPattern.md->naxis-1] <= *customPatternSliceIdx)
        {
            printf("customPatternSliceIdx is out of range for given pattern image.\n");
            return RETURN_FAILURE;
        }
        printf("Pattern stream size :");
        for (int i = 0; i < imPattern.md->naxis; i++)
            printf(" %d", imPattern.md->size[i]);
        printf("\n");
    }
    // connect to response stream if required, and check dimensions
    IMGID imResponse;
    if (*useCustomResponseStream == 1)
    {
        printf("READ RESULT: %ld\n", read_sharedmem_image(customResponseStream));
        imResponse = mkIMGID_from_name(customResponseStream);
        resolveIMGID(&imResponse, ERRMODE_ABORT);
        if (imResponse.md->naxis != imgwfs.md->naxis + 1)
        {
            printf("Response stream has wrong dimensinality. Has to be WFS stream dim +1.\n");
            return RETURN_FAILURE;
        }
        for (int i = 0; i < imgwfs.md->naxis; i++)
            if (imResponse.md->size[i] != imgwfs.md->size[i])
            {
                printf("Response stream has wrong response size. Must match WF stream size.\n");
                return RETURN_FAILURE;
            }
        if (imResponse.md->size[imResponse.md->naxis-1] <= *customResponseSliceIdx)
        {
            printf("customResponseSliceIdx is out of range for given pattern image.\n");
            return RETURN_FAILURE;
        }
        printf("Response stream size :");
        for (int i = 0; i < imResponse.md->naxis; i++)
            printf(" %d", imResponse.md->size[i]);
        printf("\n");
    }

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
        *skipMFramerate,
        *fpsMeasTime,
        *numPokes,
        *framesPerPoke,
        *saveraw,
        *pokePatternType,
        customPatternStream,
        *customPatternSliceIdx,
        *patternToStrokeMul,
        *useCustomResponseStream,
        customResponseStream,
        *customResponseSliceIdx);

    // === RECORD LATENCY
    mlsRecordDo(recorder);
   
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
