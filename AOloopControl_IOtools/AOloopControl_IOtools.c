/**
 * @file    AOloopControl_IOtools.c
 * @brief   Adaptive Optics Control loop engine I/O tools
 *
 * AO engine uses stream data structure
 *
 * @bug No known bugs.
 *
 *
 */

// module default short name
// all CLI calls to this module functions will be <shortname>.<funcname>
// if set to "", then calls use <funcname>
#define MODULE_SHORTNAME_DEFAULT "cacaoio"

// Module short description
#define MODULE_DESCRIPTION "AO loop control IO tools"

// Application to which module belongs
#define MODULE_APPLICATION "cacao"

#define _GNU_SOURCE



#include "CommandLineInterface/CLIcore.h"

#include "AOloopControl/AOloopControl.h"
#include "AOloopControl_IOtools/AOloopControl_IOtools.h"

#include "acquireWFSim.h"
#include "findspots.h"
#include "WFScamsim.h"
#include "WFSmap.h"
#include "ao188_preprocessor.h"
#include "acquireWFSspec.h"

#include "shs_on_gpu/shs_gpu_ref.h"
#include "shs_on_gpu/shs_gpu.h"
#include "shs_on_gpu/pupil_reshape.h"
#include "shs_on_gpu/pupil_zernike_gen.h"




// Module initialization macro in CLIcore.h
// macro argument defines module name for bindings
//
INIT_MODULE_LIB(AOloopControl_IOtools)







static errno_t init_module_CLI()
{
    CLIADDCMD_AOloopControl_IOtools__acquireWFSim();
    CLIADDCMD_AOloopControl_IOtools__WFScamsim();
    CLIADDCMD_AOloopControl_IOtools__WFSmap();
    CLIADDCMD_AOloopControl_IOtools__findspots();
    CLIADDCMD_AOloopControl_IOtools__AO188Preproc();
    CLIADDCMD_AOloopControl_IOtools__acquirespectra();

    // SHS evaluation on a GPU
    CLIADDCMD_AOloopControl_IOtools__recordShsRefGPU();
    CLIADDCMD_AOloopControl_IOtools__EvaluateShsGPU();
    CLIADDCMD_AOloopControl_IOtools__pupilReshape();
    CLIADDCMD_AOloopControl_IOtools__PupilZernikeGen();

    // add atexit functions here
    // atexit((void*) myfunc);

    return RETURN_SUCCESS;
}
