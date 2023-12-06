/**
 * @file    makeSimpleZonalPokeModes.c
 * @brief   Generate single actuator poke modes
 *
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"

static uint32_t *dmsizex;
static uint32_t *dmsizey;
static char *outname;

static CLICMDARGDEF farg[] =
{
    {
        CLIARG_UINT32,
        ".dmsizeX",
        "size x of the DM",
        "50",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &dmsizex,
        NULL
    },
    {
        CLIARG_UINT32,
        ".dmsizeY",
        "size y of the DM",
        "50",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &dmsizey,
        NULL
    },
    {
        CLIARG_STR_NOT_IMG,
        ".outname",
        "Name of output image",
        "Spoke",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outname,
        NULL
    },
};




static CLICMDDATA CLIcmddata =
{
    "aolmksimplezpM", "make zonal DM modes", CLICMD_FIELDS_DEFAULTS
};




static errno_t customCONFsetup()
{
    if(data.fpsptr != NULL)
    {
    }

    return RETURN_SUCCESS;
}


static errno_t customCONFcheck()
{
    if(data.fpsptr != NULL)
    {
    }
    return RETURN_SUCCESS;
}



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}

// create simple poke matrix
static imageID mkSimpleZpokeM(uint32_t dmxsize,
        uint32_t dmysize,
        char    *IDout_name)
{
    imageID  IDout;
    uint64_t dmxysize;

    dmxysize = dmxsize * dmysize;

    create_3Dimage_ID(IDout_name, dmxsize, dmysize, dmxysize, &IDout);

    for(uint64_t kk = 0; kk < dmxysize; kk++)
    {
        data.image[IDout].array.F[kk * dmxysize + kk] = 1.0;
    }

    return IDout;
}

static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT
    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        mkSimpleZpokeM(*dmsizex, *dmsizey, outname);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_milk_AOloopControl_acquireCalib__mkSimpleZonalPokeModes()
{
    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
