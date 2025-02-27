/**
 * @file    mlat.c
 * @brief   measure hardware latency
 *
 * Measure latency between DM and WFS
 *
 *
 */

#include <math.h>
#include <float.h>

#include <time.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h" // quicksort
#include "statistic/statistic.h"         // ran1()

// Local variables pointers
static char *dmstream;
long         fpi_dmstream;

static char *wfsstream;
long         fpi_wfsstream;

static float *frameratewait;
long          fpi_frameratewait;

static float *OPDamp;
long          fpi_OPDamp;

static char *pokemap;
long         fpi_pokemap;

static float *CPA;
long          fpi_CPA;

static uint32_t *NBiter;
long             fpi_NBiter;

static uint32_t *wfsNBframemax;
long             fpi_wfsNBframemax;

static float *wfsdt;
long          fpi_wfsdt;

static float *twaitus;
long          fpi_twaitus;

static float *refdtoffset;
long          fpi_refdtoffset;

static float *dtoffset;
long          fpi_dtoffset;

static float *framerateHz;
long          fpi_framerateHz;

static float *latencyfr;
long          fpi_latencyfr;

static int64_t *saveraw;
long            fpi_saveraw;

static int64_t *saveseq;
long            fpi_saveseq;

static uint32_t *seqNBframe;
long             fpi_seqNBframe;

static float *seqdtframe;
long          fpi_seqdtframe;



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
        ".OPDamp",
        "poke amplitude [um]",
        "0.1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &OPDamp,
        &fpi_OPDamp
    },
    {
        CLIARG_STREAM,
        ".pokemap",
        "optional DM poke map, use if exists",
        "null",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &pokemap,
        &fpi_pokemap
    },
    {
        CLIARG_FLOAT32,
        ".CPA",
        "Cycles/aperture [float]",
        "20",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &CPA,
        &fpi_CPA
    },
    {
        CLIARG_FLOAT32,
        ".frameratewait",
        "time period for frame rate measurement",
        "5",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &frameratewait,
        &fpi_frameratewait
    },
    {
        CLIARG_UINT32,
        ".NBiter",
        "Number of iteration",
        "100",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &NBiter,
        &fpi_NBiter
    },
    {
        CLIARG_UINT32,
        ".wfsNBframemax",
        "Number frames in measurement sequence",
        "50",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &wfsNBframemax,
        &fpi_wfsNBframemax
    },
    {
        CLIARG_FLOAT32,
        ".status.wfsdt",
        "WFS frame interval",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &wfsdt,
        &fpi_wfsdt
    },
    {
        CLIARG_FLOAT32,
        ".status.twaitus",
        "initial wait [us]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &twaitus,
        &fpi_twaitus
    },
    {
        CLIARG_FLOAT32,
        ".status.refdtoffset",
        "baseline time offset to poke",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &refdtoffset,
        &fpi_refdtoffset
    },
    {
        CLIARG_FLOAT32,
        ".status.dtoffset",
        "actual time offset to poke",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &dtoffset,
        &fpi_dtoffset
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
        ".out.latencyfr",
        "hardware latency [frame]",
        "0",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &latencyfr,
        &fpi_latencyfr
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
        CLIARG_ONOFF,
        ".option.saveseq",
        "Save sequence image cube",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &saveseq,
        &fpi_saveseq
    },
    {
        CLIARG_UINT32,
        ".option.seqNBframe",
        "Number of frames in seq cube",
        "100",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &seqNBframe,
        &fpi_seqNBframe
    },
    {
        CLIARG_FLOAT32,
        ".option.seqdtfr",
        "seq cube time resolution [fr]",
        "0.1",
        CLIARG_OUTPUT_DEFAULT,
        (void **) &seqdtframe,
        &fpi_seqdtframe
    }
};



// Optional custom configuration setup.
// Runs once at conf startup
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
    "mlat", "measure latency between DM and WFS", CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Measure latency between two streams\n");

    printf(
        "Convention\n"
        "Latency is defined here as the time offset between:\n"
        "- Time at which input stream is perturbed\n"
        "- Average arrival time between the two output frames experiencing the maximum change\n"
        "\n"
        "For example, if the output stream is a camera with full duty cycle (frame rate = 1/ exposure time),\n"
        "and there is no delay between the input and output spaces,\n"
        "then the latency will be 0.5 frame, capturing only the 1/2 frame latency inherent to the output temporal sampling\n"
    );

    return RETURN_SUCCESS;
}




static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    // connect to DM
    IMGID imgdm = mkIMGID_from_name(dmstream);
    resolveIMGID(&imgdm, ERRMODE_ABORT);
    printf("DM size : %u %u\n", imgdm.md->size[0], imgdm.md->size[1]);
    uint32_t dmxsize = imgdm.md->size[0];
    uint32_t dmysize = imgdm.md->size[1];

    // connect to WFS
    IMGID imgwfs = mkIMGID_from_name(wfsstream);
    resolveIMGID(&imgwfs, ERRMODE_ABORT);
    printf("WFS size : %u %u\n", imgwfs.md->size[0], imgwfs.md->size[1]);

    // connect to optional pokemap
    IMGID imgpokemap = mkIMGID_from_name(pokemap);
    resolveIMGID(&imgpokemap, ERRMODE_WARN);
    if(imgpokemap.ID != -1)
    {
        printf("pokemap size : %u %u\n", imgpokemap.md->size[0],
               imgpokemap.md->size[1]);
    }

    // create wfs image cube for storage
    imageID IDwfsc;
    {
        uint32_t naxes[3];
        naxes[0] = imgwfs.md->size[0];
        naxes[1] = imgwfs.md->size[1];
        naxes[2] = *wfsNBframemax;

        create_image_ID("_testwfsc",
                        3,
                        naxes,
                        imgwfs.datatype,
                        0,
                        0,
                        0,
                        &IDwfsc);
    }

    float *latencyarray = (float *) malloc(sizeof(float) * *NBiter);
    if(latencyarray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort(); // or handle error in other ways
    }

    float *latencysteparray = (float *) malloc(sizeof(float) * *NBiter);
    if(latencysteparray == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }


    // Prepare output diff cube sequence
    // if saveraw = ON
    //
    long diffseqsize = *seqNBframe; // number of slices
    float diffseqdtframe = *seqdtframe; // time increment per slice [frame]
    float   *diffseqkkcnt = (float *) malloc(sizeof(float) *
                            diffseqsize); // count how many frames go in each slice
    for(int diffseqkk = 0; diffseqkk < diffseqsize; diffseqkk++)
    {
        diffseqkkcnt[diffseqkk] = 0.0;
    }
    IMGID imgdiffseq;
    imgdiffseq = makeIMGID_3D("CMmodesWFS",
                              imgwfs.md->size[0],
                              imgwfs.md->size[1],
                              diffseqsize);
    createimagefromIMGID(&imgdiffseq);




    // Create DM patterns
    imageID IDdm0 = -1;
    imageID IDdm1 = -1;
    {
        create_2Dimage_ID("_mlattestdm0", dmxsize, dmysize, &IDdm0);
        create_2Dimage_ID("_mlattestdm", dmxsize, dmysize, &IDdm1);

        float RMStot = 0.0;
        for(uint32_t ii = 0; ii < dmxsize; ii++)
            for(uint32_t jj = 0; jj < dmysize; jj++)
            {
                float x = (2.0 * ii - 1.0 * dmxsize) / dmxsize;
                float y = (2.0 * jj - 1.0 * dmxsize) / dmysize;
                data.image[IDdm0].array.F[jj * dmxsize + ii] = 0.0;
                data.image[IDdm1].array.F[jj * dmxsize + ii] =
                    (*OPDamp) * (sin(*CPA * x) * sin(*CPA * y));
                RMStot += data.image[IDdm1].array.F[jj * dmxsize + ii] *
                          data.image[IDdm1].array.F[jj * dmxsize + ii];
            }
        RMStot = sqrt(RMStot / dmxsize / dmysize);

        printf("RMStot = %f\n", RMStot);

        for(uint32_t ii = 0; ii < dmxsize; ii++)
            for(uint32_t jj = 0; jj < dmysize; jj++)
            {
                data.image[IDdm1].array.F[jj * dmxsize + ii] *=
                    (*OPDamp) / RMStot;
            }

        // save output
        fps_write_RUNoutput_image(data.fpsptr, "_mlattestdm", "mlatpokeDM");
    }





    INSERT_STD_PROCINFO_COMPUTEFUNC_START
    {
        // Measure coarse estimate of frame rate
        //
        int framerateOK = 0;
        {
            double          tdouble_start;
            double          tdouble_end;
            long            wfscntstart;
            long            wfscntend;
            struct timespec tnow;

            long stringmaxlen = 200;
            char msgstring[stringmaxlen];
            snprintf(msgstring,
                     stringmaxlen,
                     "Measuring approx frame rate over %.1f sec",
                     *frameratewait);
            processinfo_WriteMessage(processinfo, msgstring);

            clock_gettime(CLOCK_MILK, &tnow);
            tdouble_start = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;
            wfscntstart   = imgwfs.md->cnt0;

            {
                long nsec           = (long)(1000000000 * (*frameratewait));
                long nsec_remaining = nsec % 1000000000;
                long sec            = nsec / 1000000000;

                struct timespec timesleep;
                timesleep.tv_sec  = sec;
                timesleep.tv_nsec = nsec_remaining;

                nanosleep(&timesleep, NULL);
            }

            //        usleep( (long)  (1000000 * (*frameratewait)) );

            clock_gettime(CLOCK_MILK, &tnow);
            tdouble_end = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;
            wfscntend   = imgwfs.md->cnt0;
            *wfsdt      = (tdouble_end - tdouble_start) / (wfscntend - wfscntstart);

            printf("wfs dt = %f sec\n", *wfsdt);

            if(wfscntend - wfscntstart < 5)
            {
                snprintf(msgstring,
                         stringmaxlen,
                         "Number of frames %ld too small -> cannot proceed",
                         wfscntend - wfscntstart);
                processinfo_error(processinfo, msgstring);
                printf("%s\n", msgstring);
            }
            else
            {
                framerateOK = 1;
                snprintf(msgstring,
                         stringmaxlen,
                         "frame period wfsdt = %f sec  ( %f Hz )\n",
                         *wfsdt,
                         1.0 / *wfsdt);
                processinfo_WriteMessage(processinfo, msgstring);

                // This is approximate, will be measured more precisely later on
                *framerateHz = 1.0 / (*wfsdt);
            }
        }


        // Measure latency
        //
        if(framerateOK == 1)
        {
            // store all raw measurements in these arrays
            //
            double *valarrayall = (double *) malloc(sizeof(double) * (*wfsNBframemax) *
                                                    (*NBiter));
            double *dtarrayall  = (double *) malloc(sizeof(double) * (*wfsNBframemax) *
                                                    (*NBiter));

            // number of raw measurements
            int NBrawmeas = 0;


            // Measure latency
            double tdouble_start;
            double tdouble_end;

            double dtmax; // Max running time per iteration

            // update timing parameters for poke test
            dtmax        = *wfsdt * (*wfsNBframemax) * 1.2 + 0.5;
            *twaitus     = 1000000.0 * *wfsdt * 3.0; // wait 3 frames
            *refdtoffset = 0.5 * *wfsdt;

            struct timespec *tarray;
            tarray = (struct timespec *) malloc(sizeof(struct timespec) *
                                                (*wfsNBframemax));
            if(tarray == NULL)
            {
                PRINT_ERROR("malloc returns NULL pointer");
                abort(); // or handle error in other ways
            }

            double *dtarray;
            dtarray = (double *) malloc(sizeof(double) * (*wfsNBframemax));
            if(dtarray == NULL)
            {
                PRINT_ERROR("malloc returns NULL pointer");
                abort(); // or handle error in other ways
            }

            FILE *fphwlat =
                fps_write_RUNoutput_file(data.fpsptr, "hardwlatency", "dat");

            struct timespec tnow;
            clock_gettime(CLOCK_MILK, &tnow);
            tdouble_start       = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;
            long wfscntstart    = imgwfs.md->cnt0;
            long wfsframeoffset = (long)(0.1 * (*wfsNBframemax));

            // Measurement loop

            uint32_t iter   = 0;
            int      loopOK = 1;

            // RECORD SEQUENCES START
            while(loopOK == 1)
            {
                double        tstartdouble;
                long          NBwfsframe;
                unsigned long wfscnt0;
                double        latencymax = 0.0;
                double        latency;


                uint64_t wfssize = imgwfs.md->size[0] * imgwfs.md->size[1];

                struct timespec tstart;
                long            kkoffset = 0;
                long            kkmax    = 0;


                processinfo_WriteMessage_fmt(processinfo,
                                             "iteration %5u / %5u",
                                             iter,
                                             *NBiter);



                for(uint32_t ii = 0; ii < dmxsize * dmysize; ii++)
                {
                    imgdm.im->array.F[ii] = 0.0;
                }
                processinfo_update_output_stream(processinfo, imgdm.ID);







                unsigned int dmstate = 0;

                // Waiting for streams to settle
                // Wait time is a few frames
                //
                {
                    long nsec = (long)(1000 * (*twaitus));

                    long nsec_remaining = nsec % 1000000000;
                    long sec            = nsec / 1000000000;

                    struct timespec timesleep;
                    timesleep.tv_sec  = sec;
                    timesleep.tv_nsec = nsec_remaining;

                    nanosleep(&timesleep, NULL);
                }

                // Waiting for a number of frames for streams to settle
                //
                wfscnt0 = imgwfs.md->cnt0;
                for(uint32_t wfsframe = 0; wfsframe < *wfsNBframemax; wfsframe++)
                {
                    while(wfscnt0 == imgwfs.md->cnt0)
                    {
                        long nsec = (long)(1000 * 50);  // 50 usec

                        long nsec_remaining = nsec % 1000000000;
                        long sec            = nsec / 1000000000;

                        struct timespec timesleep;
                        timesleep.tv_sec  = sec;
                        timesleep.tv_nsec = nsec_remaining;

                        nanosleep(&timesleep, NULL);
                    }
                    wfscnt0 = imgwfs.md->cnt0;
                }


                // Set timer reference point
                //
                double dt = 0.0;
                clock_gettime(CLOCK_MILK, &tstart);
                tstartdouble = 1.0 * tstart.tv_sec + 1.0e-9 * tstart.tv_nsec;


                long wfsframe = 0;
                int  wfsslice = 0;
                wfscnt0       = imgwfs.md->cnt0;

                // RECORD FRAMES OF A SEQUENCE START
                while((dt < dtmax) && (wfsframe < *wfsNBframemax))
                {
                    // WAITING for new image
                    //
                    while(wfscnt0 == imgwfs.md->cnt0)
                    {
                        long nsec = (long)(1000 * 2);  // 2 usec

                        long nsec_remaining = nsec % 1000000000;
                        long sec            = nsec / 1000000000;

                        struct timespec timesleep;
                        timesleep.tv_sec  = sec;
                        timesleep.tv_nsec = nsec_remaining;

                        nanosleep(&timesleep, NULL);
                    }
                    wfscnt0 = imgwfs.md->cnt0;


                    // Copy output (WFS) image to storage cube, into slide # wfsframe
                    //
                    wfsslice = 0;
                    int   datatype_size = ImageStreamIO_typesize(imgwfs.datatype);
                    char *ptr0          = ImageStreamIO_get_image_d_ptr(imgwfs.im);
                    ptr0 += datatype_size * wfsslice * wfssize;

                    char *ptr = ImageStreamIO_get_image_d_ptr(&data.image[IDwfsc]);
                    ptr += datatype_size * wfsframe * wfssize;

                    memcpy(ptr, ptr0, datatype_size * wfssize);

                    // Record time
                    // store in dtarray
                    //
                    clock_gettime(CLOCK_MILK, &tarray[wfsframe]);

                    double tdouble = 1.0 * tarray[wfsframe].tv_sec +
                                     1.0e-9 * tarray[wfsframe].tv_nsec;
                    dt = tdouble - tstartdouble;
                    dtarray[wfsframe] = dt;




                    // At roughly the half time, apply DM pattern #1
                    // This is only done once per sequence, using dmstate toggle from 0 to 1
                    //
                    if((dmstate == 0) && (dt > *refdtoffset) &&
                            (wfsframe > wfsframeoffset))
                    {
                        // Add a random wait to obtain continuous time offset sampling
                        //
                        {
                            long nsec = (long)(1000000000.0 * ran1() * (*wfsdt));

                            long nsec_remaining = nsec % 1000000000;
                            long sec            = nsec / 1000000000;

                            struct timespec timesleep;
                            timesleep.tv_sec  = sec;
                            timesleep.tv_nsec = nsec_remaining;

                            nanosleep(&timesleep, NULL);
                        }

                        kkoffset = wfsframe;

                        dmstate = 1;
                        if(imgpokemap.ID == -1)
                        {
                            copy_image_ID("_mlattestdm", dmstream, 1);
                        }
                        else
                        {
                            for(uint32_t ii = 0; ii < dmxsize * dmysize; ii++)
                            {
                                imgdm.im->array.F[ii] = (*OPDamp) * imgpokemap.im->array.F[ii];
                            }
                            processinfo_update_output_stream(processinfo, imgdm.ID);
                        }

                        // Record time at which DM command is sent
                        //
                        clock_gettime(CLOCK_MILK, &tnow);
                        tdouble   = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;
                        dt        = tdouble - tstartdouble;
                        *dtoffset = dt; // time at which DM command is sent
                    }
                    wfsframe++;
                }


                if(imgpokemap.ID == -1)
                {
                    copy_image_ID("_mlattestdm", dmstream, 1);
                }
                else
                {
                    for(uint32_t ii = 0; ii < dmxsize * dmysize; ii++)
                    {
                        imgdm.im->array.F[ii] = (*OPDamp) * imgpokemap.im->array.F[ii];
                    }
                    processinfo_update_output_stream(processinfo, imgdm.ID);
                }
                dmstate = 0;


                if(data.fpsptr->parray[fpi_saveraw].fpflag & FPFLAG_ONOFF)
                {
                    // Save each datacube
                    //
                    char ffnameC[STRINGMAXLEN_FULLFILENAME];
                    WRITE_FULLFILENAME(ffnameC,
                                       "mlat-testC-%04d", iter);
                    fps_write_RUNoutput_image(data.fpsptr, "_testwfsc", ffnameC);
                }

                // Computing difference between consecutive images
                NBwfsframe = wfsframe;

                double *valarray = (double *) malloc(sizeof(double) * NBwfsframe);
                if(valarray == NULL)
                {
                    PRINT_ERROR("malloc returns NULL pointer");
                    abort();
                }

                double valmax   = 0.0;
                double valmaxdt = 0.0;
                double valmin = DBL_MAX;


                float *diffseqvalarray = (float *) malloc(sizeof(float) * wfssize);
                // Measure latency from stored image cube
                // For each time step (= slice in cube), measure magnitude of change
                // between current and previous frame.
                // Store result in valarray
                // Scan valarry looking for maximum -> store in valmax
                //
                // Define a macro for the type switching that follows
#define IMAGE_SUMMING_CASE(IMG_PTR_ID)                                             \
    {                                                                              \
        for (uint64_t ii = 0; ii < wfssize; ii++)                                  \
        {                                                                          \
            double tmp =                                                           \
                1.0*data.image[IDwfsc].array.IMG_PTR_ID[kk * wfssize + ii] -       \
                1.0*data.image[IDwfsc].array.IMG_PTR_ID[(kk - 1) * wfssize + ii];  \
            valarray[kk] += 1.0 * tmp * tmp;                                       \
            diffseqvalarray[ii] = tmp;                                             \
        }                                                                          \
    }


                for(long kk = 1; kk < NBwfsframe; kk++)
                {
                    valarray[kk] = 0.0;

                    switch(imgwfs.datatype)
                    {
                    case _DATATYPE_FLOAT:
                        IMAGE_SUMMING_CASE(F);
                        break;
                    case _DATATYPE_DOUBLE:
                        IMAGE_SUMMING_CASE(D);
                        break;
                    case _DATATYPE_UINT16:
                        IMAGE_SUMMING_CASE(UI16);
                        break;
                    case _DATATYPE_INT16:
                        IMAGE_SUMMING_CASE(SI16);
                        break;
                    case _DATATYPE_UINT32:
                        IMAGE_SUMMING_CASE(UI32);
                        break;
                    case _DATATYPE_INT32:
                        IMAGE_SUMMING_CASE(SI32);
                        break;
                    case _DATATYPE_UINT64:
                        IMAGE_SUMMING_CASE(UI64);
                        break;
                    case _DATATYPE_INT64:
                        IMAGE_SUMMING_CASE(SI64);
                        break;
                    case _DATATYPE_COMPLEX_FLOAT:
                    case _DATATYPE_COMPLEX_DOUBLE:
                    default:
                        PRINT_ERROR("COMPLEX TYPES UNSUPPORTED");
                        return RETURN_FAILURE;
                    }

                    valarray[kk] = sqrt(valarray[kk] / wfssize / 2);

                    {
                        float timeoffset = (0.5 * (dtarray[kk] + dtarray[kk - 1]) - *dtoffset) *
                                           (*framerateHz);
                        int diffseqkk = timeoffset / diffseqdtframe;
                        if((diffseqkk >= 0) && (diffseqkk < diffseqsize))
                        {
                            for(uint64_t ii = 0; ii < wfssize; ii++)
                            {
                                imgdiffseq.im->array.F[diffseqkk * wfssize + ii] += diffseqvalarray[ii];
                            }
                            diffseqkkcnt[diffseqkk] += 1.0;
                        }
                    }



                    // Look for maximum change between frames
                    //
                    // CONVENTION:
                    // Time of max change (valmaxdt) is midpoint between arrival (read) time of frames k and k-1
                    //
                    if(valarray[kk] > valmax)
                    {
                        valmax   = valarray[kk];
                        valmaxdt = 0.5 * (dtarray[kk - 1] + dtarray[kk]);
                        kkmax    = kk - kkoffset;
                    }
                }

                free(diffseqvalarray);

                //
                //
                //
                for(wfsframe = 1; wfsframe < NBwfsframe; wfsframe++)
                {
                    double ptdt = (0.5 * (dtarray[wfsframe] + dtarray[wfsframe - 1]) - *dtoffset);
                    double ptval = valarray[wfsframe];

                    valarrayall[NBrawmeas] = ptval;
                    dtarrayall[NBrawmeas] = ptdt;
                    NBrawmeas++;

                    fprintf(fphwlat,
                            "%ld   %10.2f     %g\n",
                            wfsframe - kkoffset,
                            1.0e6 * (0.5 * (dtarray[wfsframe] + dtarray[wfsframe - 1]) - *dtoffset),
                            valarray[wfsframe]);
                }


                free(valarray);

                latency = valmaxdt - *dtoffset;
                // latencystep = kkmax;


                if(latency > latencymax)
                {
                    latencymax = latency;
                }

                fprintf(fphwlat, "# %5u  %8.6f\n", iter, (valmaxdt - *dtoffset));

                latencysteparray[iter] = 1.0 * kkmax;
                latencyarray[iter]     = (valmaxdt - *dtoffset);

                // process signals, increment loop counter
                iter++;
                if(iter == (*NBiter))
                {
                    loopOK = 0;
                }
            } // RECORD SEQUENCES END

            fclose(fphwlat);

            clock_gettime(CLOCK_MILK, &tnow);
            tdouble_end    = 1.0 * tnow.tv_sec + 1.0e-9 * tnow.tv_nsec;
            long wfscntend = imgwfs.md->cnt0;

            free(tarray);
            free(dtarray);



            // PROCESS RESULTS
            //
            processinfo_WriteMessage_fmt(processinfo, "Processing Data (%u iterations)",
                                         (*NBiter));


            // normalize imgdiffseq
            {
                uint64_t wfssize = imgwfs.md->size[0] * imgwfs.md->size[1];
                for(int diffseqkk = 0; diffseqkk < diffseqsize; diffseqkk++)
                {
                    if(diffseqkkcnt[diffseqkk] > 0.5)
                    {
                        for(uint64_t ii = 0; ii < wfssize; ii++)
                        {
                            imgdiffseq.im->array.F[diffseqkk * wfssize + ii] /= diffseqkkcnt[diffseqkk];
                        }
                    }
                }
            }
            // Save imgdiffseq
            //
            if(data.fpsptr->parray[fpi_saveseq].fpflag & FPFLAG_ONOFF)
            {
                char ffnameC[STRINGMAXLEN_FULLFILENAME];
                WRITE_FULLFILENAME(ffnameC,
                                   "mlat-diffseq");
                fps_write_RUNoutput_image(data.fpsptr, imgdiffseq.name, ffnameC);
            }




            copy_image_ID("_mlattestdm0", dmstream, 1);

            float latencyave     = 0.0;
            float latencystepave = 0.0;
            float minlatency     = latencyarray[0];
            float maxlatency     = latencyarray[0];
            for(uint32_t iter = 0; iter < (*NBiter); iter++)
            {
                if(latencyarray[iter] > maxlatency)
                {
                    maxlatency = latencyarray[iter];
                }

                if(latencyarray[iter] < minlatency)
                {
                    minlatency = latencyarray[iter];
                }

                latencyave += latencyarray[iter];
                latencystepave += latencysteparray[iter];
            }
            latencyave /= (*NBiter);
            latencystepave /= (*NBiter);

            // measure precise frame rate
            //
            double dt = tdouble_end - tdouble_start;
            printf("FRAME RATE = %.3f Hz\n", 1.0 * (wfscntend - wfscntstart) / dt);
            *framerateHz = 1.0 * (wfscntend - wfscntstart) / dt;
            functionparameter_SaveParam2disk(data.fpsptr, ".out.framerateHz");




            // Detect peak using all points
            //
            double latencymeaspeakdt = 0.0;
            double latencymeaspeakval = 0.0;
            quick_sort2(dtarrayall, valarrayall, NBrawmeas);

            {
                FILE *fpout;
                fpout = fps_write_RUNoutput_file(data.fpsptr, "hardwlatencypts", "dat");
                int iimin = 0;
                int iimax = 0;
                float dtrange = 0.5 / (*framerateHz); // in sec
                for(int ii = 0; ii < NBrawmeas; ii++)
                {
                    while((dtarrayall[iimax] < dtarrayall[ii] + dtrange) && (iimax < NBrawmeas - 1))
                    {
                        iimax ++;
                    }
                    while((dtarrayall[iimin] < dtarrayall[ii] - dtrange) && (iimax < NBrawmeas - 1))
                    {
                        iimin ++;
                    }

                    long nbpts = iimax - iimin;
                    double dtmedian = dtarrayall[ii];
                    double valmedian = valarrayall[ii];
                    if(nbpts > 0)
                    {
                        // Take average of median 1/3 values

                        double *ptsval = (double *) malloc(sizeof(double) * nbpts);
                        double *ptsdt = (double *) malloc(sizeof(double) * nbpts);
                        for(int jj = 0; jj < nbpts; jj++)
                        {
                            ptsval[jj] = valarrayall[iimin + jj];
                            ptsdt[jj] = dtarrayall[iimin + jj];
                        }
                        quick_sort2(ptsval, ptsdt, nbpts);
                        double dtave = 0.0;
                        double valave = 0.0;
                        double cave = 0.0;

                        for(int jj = nbpts / 3; jj < 2 * nbpts / 3; jj++)
                        {
                            double coeff = 1.0;
                            dtave += coeff * ptsdt[jj];
                            valave += coeff * ptsval[jj];
                            cave += coeff;
                        }
                        dtave /= cave;
                        valave /= cave;

                        dtmedian = dtave;
                        valmedian = valave;

                        free(ptsval);
                        free(ptsdt);
                    }

                    if(valmedian > latencymeaspeakval)
                    {
                        latencymeaspeakval = valmedian;
                        latencymeaspeakdt = dtmedian;
                    }

                    fprintf(fpout, "%d %g %g %d %d %g %g  %g %g %ld\n",
                            ii, dtarrayall[ii], valarrayall[ii], iimin, iimax, dtarrayall[iimin],
                            dtarrayall[iimax],
                            dtmedian, valmedian, nbpts);

                }

                fclose(fpout);
            }

            free(dtarrayall);
            free(valarrayall);

            printf("latency peak at %g, value %g\n", latencymeaspeakdt, latencymeaspeakval);


            // update latencystepave from framerate


            *latencyfr = latencymeaspeakdt * (*framerateHz);
            printf("latency = %f frame\n", *latencyfr);
            functionparameter_SaveParam2disk(data.fpsptr, ".out.latencyfr");

            {
                FILE *fpout;
                fpout =
                    fps_write_RUNoutput_file(data.fpsptr, "param_hardwlatency", "txt");
                fprintf(fpout, "%8.6f", 1.01);
                fclose(fpout);
            }




            // write results as env variables
            {
                // file will be sourced by cacao-check-cacaovars
                //
                char ffname[STRINGMAXLEN_FULLFILENAME];
                WRITE_FULLFILENAME(ffname, "%s/cacaovars.bash", data.fpsptr->md->datadir);

                printf("SAVING TO %s\n", ffname);

                FILE *fpout;
                fpout = fopen(ffname, "w");
                fprintf(fpout, "export CACAO_WFSFRATE=%.3f\n", *framerateHz);
                fprintf(fpout, "export CACAO_LATENCYHARDWFR=%.3f\n", *latencyfr);
                fclose(fpout);
            }

        }
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    free(latencyarray);
    free(latencysteparray);

    free(diffseqkkcnt);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


INSERT_STD_FPSCLIfunctions


// Register function in CLI
errno_t
CLIADDCMD_AOloopControl_perfTest__mlat()
{

    CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
