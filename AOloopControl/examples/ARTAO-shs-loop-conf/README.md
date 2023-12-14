This readme-file is a WIP file. It is a copy of the [[scexao-readme]] file and will be changed step by step to introduce the user to using the ARTAO example.

# Overview

Example of the Apollon Real Time Adaptive Optics (ARTAO) system.
An AO system, currently developed at the ultra-intense Apollon laser system in France for focal spot stabilization, using a fast Shack-Hartmann Sensor (SHS), evaluated on a GPU.

This is a WIP example and may change several times until a stable state is reached. Also, as for now, no simulation mode is available.


# Running the example

:warning: Check the [instructions](https://github.com/cacao-org/cacao/tree/dev/AOloopControl/examples) before running these steps

## 1. Setting up processes
```bash
# Deploy configuration :
# download from source to current directory
cacao-loop-deploy -c ARTAO-shs-loop

# OPTIONAL: Change loop number, name, DM index, simulation DM index:
# CACAO_LOOPNUMBER=8 cacao-loop-deploy -c ARTAO-shs-loop
# CACAO_LOOPNUMBER=8 CACAO_DMINDEX="01" cacao-loop-deploy -c ARTAO-shs-loop

# OPTIONAL: Edit file ARTAO-shs-loop-conf/cacaovars.bash as needed
# For example, change loop index, DM index, etc ...

# OPTIONAL: Clean previous deployment :
# rm -rf .artao-loop.cacaotaskmanager-log


# Run deployment (starts conf processes)
cacao-loop-deploy -r ARTAO-shs-loop

# Note: the copy and run steps can be done at once with :
# cacao-loop-deploy ARTAO-shs-loop


# Go to rootdir, from which user controls the loop
cd artao-loop-rootdir

```

### Logging and fpsCTRL start

Deploy logging processes and terminals. Upon these commandsm, four terminals will open on the current desktop environment:
- ``fpsCTRLlog-artao-loop``: display logging output of fpsCTRL commands
- ``Operatorlog-artao-loop-output``: display log entries done by cacao scripts or the operators
- ``Operatorlog-artao-loop-input``: custom log entries can be made here for later reference - e.g. actions like switching reference or noreworthy events during operation.
- ``Weatherlog-artao-loop-input``: a remnant of astronomy AO. Feel free to write about the weather outside of the lab or close it using CTRL+C in the terminal.
- 
```bash
cacao-msglogCTRL start
cacao-msglogCTRL terms
```
Note: at the end of an operation day, run ``cacao-msglogCTRL stop`` to mark the end in the logfiles and close the terminals. The next day, the logging can be started again using the commands above.

Start fpsCTRL terminal:
```bash
cacao-fpsctrl-TUI
```
Alternatively, feel free to run ``milk-fpsCTRL`` anywhere you like.

## 2. Start the image acquisition and DM
The cacao-internal parts of the ARTAO loop assumes that
- the SHS camera image will be availabe in a stream named ``aolx-shsCam``.
- the DM is controlled by the ``aolx_dmdisp`` stream.

The ``aolx-shsCam`` stream needs to be created by the user.

The ``aolx_dmdisp`` stream is created by the dmdisp milk process, which can be launched using a premade cacao script:
```bash
# Start dm comb process
cacao-aorun-000-dm start
```


### 2.1 Start hardware controllers
The hardware controllers of the original ARTAO setup are launched with a user script:
```bash
# Start hardware controllers.
# If other hardware shall be used, this is the script to change.
./scripts/aorun-setmode-hardw
```

Right now, ARTAO uses a XIMEA PCIe camera as the SHS cam, which will be fed into ImageStreamIO using a [camera control process](https://github.com/jbOhland93/ximea-xib-64-2-milk).
Furthermore, ARTAO uses a Dynamic Optics DM, which will be fed the output via a corresponding [DM control process](https://github.com/jbOhland93/milk-2-dynamic-optics-dm).
If the same hardware is used, these have to be downloaded and built prior to the execution of the ``aorun-setmode-hardw`` script.

Note: A simulation mode is not implemented as this state, but may be added in the future.

### 2.2 Start image acquisition
```bash
# Attach to the camera control terminal
tmux a -t shsCam-X
```

```bash
# XIMEA CAMERA CONTROL APPLICATION
# Display control options
help

# Start the acquisition
start

# Set the FPS of the acquisition
setFPS 3900

# Keep this tmux session open for camera control or detach using ctrl+b, d.
```

### 2.3 Relax DM (optional)
```bash
# Attach to the DM control terminal
tmux a -t DO_DM-X
```

```bash
# DM CONTROLL APPLICATION
# Display control options
help

# Disarm the DM (stop listening to ISIO output)
disarm

# Run the relax routine
relax

# Arm the DM (listen to ISIO output again)
arm

# Keep this tmux session open for camera control or detach using ctrl+b, d.
```


## 3. Measure SHS camera dark
Block the beam towards the SHS camera, then record the average of the stream.
```bash
# Average aolX_shsCam, store in aolX_shsCam_AVG and save/log as fits file.
./scripts/aorun-001-take-shs-avg -n 2000
```
Then, unblock the beam again.

## 4. Record SHS reference
Make sure that the beam is unblocked and that the spots on the SHS camera are well iluminated. Then, run the calibration script:
```bash
# Record a reference for the SHS evaluation on GUP. -n is the number of frames used.
./scripts/aorun-002-take-shs-ref -n 64
```

## 5. Launch SHS evaluation
With the reference recorded and in SHM, the evaluation process can now be launched
```bash
# Start SHS evaluation process
./scripts/aorun-003-shs-eval start
```

The evaluation routine generates 1D data. For visual inspection, the data has to be reordered according to the reference mask.
```bash
# Start reshaping process for the wavefront output
# Reshapes aolX_shsEval_wfOut to aolX_shsEval_wfOut_reshape
./scripts/aorun-004-reshape-to-wfs-pupil shsWf start
```
Note: Other options besides ``shsWf`` include ``shsGrad`` for the gradient output and ``shsInt`` for the intensity output. However, these outputs are not served by the evaluation process by default as the process of copying data from the GPU into host memory takes time. Therefore, they have to be turned on in the FPS of the shs evaluation process manually (shsOnGPU-Eval-x.comp.cpyGradToCPU and shsOnGPU-Eval-x.comp.cpyIntToCPU). This is not done by the reshaping script as the reshaping is solely for monitoring purpose and shall not change the performance of the loop and the user should turn on these options conciously, if desired.

## 6. Start WF referencing
During operation, changing the zero point offset in WFS space can be useful, e.g. for focal spot optimization or zero-point setting. In this example, we are abusing the wavefront sensor acquisition function that usually processes pyramid WFS data into a format that can be used for AO loop control. As we already have a fully reconstructed WF that can be used as linear input, the dark subtraction and normalization are turned off in the fpssetup.setval.conf file.
```bash
# Start the WFS acquisiton process, which serves as WF manipulator in this example
cacao-aorun-025-acqWFS -w start
```

### 6.1 Referencing
This process can also be used to set the current WF as reference point. For this, a fixed number of WF samples is collected and then immediately prepared to be subtracted from each measured WF.
```bash
# Acquire WFS reference
cacao-aorun-026-takeref -n 2000
# Reshape the result for visual inspection (aolX_imWFS2_reshape)
./scripts/aorun-004-reshape-to-wfs-pupil acqWfC start
# Reshape the temporal moving average for visual inspection (aolX_imWFS3_reshape)
./scripts/aorun-004-reshape-to-wfs-pupil acqWfAVG start
# NOTE: The reference is NOT subtracted automatically.
# To observe the change, enter milk-fpsCTRL, select the
# acqWFS-X process and enable the comp.WFSrefsub parameter.
```

## 7. Measure DM to WFS latency
Note that the fps of the latency measurement, ``mlat-x``, has ``.option.slowDM`` enabled as the bimorph DM features a rise time of ~1 ms, which is longer than the frame duration of the acquisition process. This value is automatically set according to the `fpssetup.setval.conf` file.
```bash
# Measure latency
cacao-aorun-021-mlat-slowDM -w
```

## 8. Acquire Calibration
In this section, the response matrix of the DM is recorded and converted into a control matrix. This process is devided into several steps.


### 8.1. Prepare DM poke modes
Instead of indexing through the actuators of the DM by default, cacao probes the DM using pre-defined mode sets. These *can* be single acutator pokes, but depending on the DM type, there may be more desirable modes. Especially for DMs with a localized actuator response, like MEMS DMs, the hadamard modes are a good choice as the signal for each poke is maximized.
The modes are pre-generated into fits-files for better performance.

```bash
# Create DM poke mode cubes
cacao-mkDMpokemodes
```
The following files are written to ./conf/RMmodesDM/
| File                 | Contents                                            |
| -------------------- | --------------------------------------------------- |
| `DMmask.fits     `   | DM mask                                             |
| `FpokesC.<CPA>.fits` | Fourier modes (where \<CPA> is an integer)          |
| `ZpokesC.<NUM>.fits` | Zernike modes (where \<NUM> is the number of modes) |
| `HpokeC.fits     `   | Hadamard modes                                      |
| `Hmat.fits       `   | Hadamard matrix (to convert Hadamard-zonal)         |
| `Hpixindex.fits  `   | Hadamard pixel index                                |
| `SmodesC.fits    `   | *Simple* (single actuator) pokes                    |

### 8.2. Acquire WFS response
In this example, we are going to use single actuator pokes instead of the more advanced Hadamard pokes as used by the SCExAO examples. This is because of two reasons related to the bimorph DM of ARTAO:
1. The response functions of the bimorph DM actuators are not localized like the ones of MEMS DMs but distributed over the whole aperture. This means that the SNR is inherently better for single actuator pokes than for in the case of a MEMS DM. Therefore, ARTAO does not profit from Hadamard poking.
2. The hysteresis of the piezos contributes to the response matrix in a deterministic manner when using single actuator pokes, while this is uncertain for Hadamard modes due to the irregular control sequence for each actuator.

```bash
# Acquire response matrix - Single actuator modes
# 10 cycles
cacao-aorun-030-acqlinResp -n 10 SmodesC
```
This could take a while. Check status on milk-procCTRL and/or watch the wavefront evolution. To inspect results, reshape the file conf/RMmodesWFS/HpokeC.WFSresp.fits:
```bash
# Reshape the original fits file to the pupil
./scripts/aorun-005-reshape-fits-to-wfs-pupil conf/RMmodesWFS/SmodesC.WFSresp.fits
# Inspect the output file conf/RMmodesWFS/SmodesC.WFSresp_rshp.fits
```

### 8.3. Compute control matrix (straight)
The computation for the control matrix expects mask fits files, which were not generated to that point. The WFS masking technically happens during referencing, but the file is still required for the calculation. The following script can be used to generate the masks:
```bash
cacao-aorun-032-RMmkmask -f conf/RMmodesWFS/RMmodesWFS.fits
```

Now, compute the control modes, in both WFS and DM spaces.
Set GPU device (if GPU available).

```bash
cacao-fpsctrl setval compstrCM svdlim 0.005
cacao-fpsctrl setval compstrCM GPUdevice 0
```
Then run the compstrCM process to compute CM and load it to shared memory :
```bash
cacao-aorun-039-compstrCM
```
The results are written to:
- conf/CMmodesDM/CMmodesDM.fits
- conf/CMmodesWFS/CMmodesWFS.fits
```bash
# Reshape the WF modes to visualize the result:
./scripts/aorun-005-reshape-fits-to-wfs-pupil conf/CMmodesWFS/CMmodesWFS.fits
# Inspect the output file conf/CMmodesWFS/CMmodesWFS_rshp.fits
```


## 9. Running the loop

### 9.1. Core processes

Select GPUs for the modal decomposition (WFS->modes) and expansion (modes->DM) MVMs
```bash
cacao-fpsctrl setval wfs2cmodeval GPUindex 0
cacao-fpsctrl setval mvalC2dm GPUindex 0
```
Start the 3 control loop processes :

```bash
# start WFS -> mode coefficient values
cacao-aorun-050-wfs2cmval start

# start modal filtering
cacao-aorun-060-mfilt start

# start mode coeff values -> DM
cacao-aorun-070-cmval2dm start

```

### 9.2 Closing the loop and setting loop parameters with mfilt:

```bash
# Set loop gain
cacao-fpsctrl setval mfilt loopgain 0.081
# Set loop mult
cacao-fpsctrl setval mfilt loopmult 0.999

# set modal gains, mults and limits
cacao-aorun-061-setmgains 0.3 -f 0.7 -t 1.4
cacao-aorun-062-setmmults 0.05 -f 0.9 -t 1.0
cacao-aorun-063-setmlimits 0.8 -f 0.05 -t 1.0

# close loop
cacao-fpsctrl setval mfilt loopON ON
```

Misc tools:
```bash
# Set WFS reference to flat illumination over wfsmask
cacao-wfsref-setflat
```


### 9.3 Zero Point Offsetting using Zernike Polynomials
A common way to change the shape of the focal spot is to apply a known amount of Zernike aberrations to the beam. As the DM of the ARTAO system does not feature a regular grid of actuators, this is most easily done in WFS space. In this example, we are applying Zernike aberrations to the ZPO stream of the WFS acquisition process.
```bash
# Check the available Zernike polynomials of the script
./scripts/aorun-010-add-zernike-wfs-offset -h
# Example: apply a bit of defocusing and some negative Coma Y
# The -r flag will provide an additional 2D representation for visual inspection (aolX_wfszpo_reshape)
./scripts/aorun-010-add-zernike-wfs-offset set defoc 0.3 -r
./scripts/aorun-010-add-zernike-wfs-offset setneg comay 0.2 -r
# Observe the change in the corrected WF (aolX_imWFS2_reshape)
./scripts/aorun-004-reshape-to-wfs-pupil acqWfC start
```

## 9. Testing the loop

### 9.1. SelfRM

```bash
# Set max number of modes above nbmodes to measure all modes

cacao-fpsctrl runstop mfilt 0 0
cacao-fpsctrl setval mfilt selfRM.zsize 100
cacao-fpsctrl setval mfilt selfRM.NBmode 100
cacao-fpsctrl runstart mfilt 0 0
cacao-fpsctrl setval mfilt selfRM.enable ON
```

Check result: artao-loop-rundir/selfRM.fits

### 9.2. Turbulence

```bash
cacao-aorun-100-DMturb start
cacao-aorun-100-DMturb off
cacao-aorun-100-DMturb on
cacao-aorun-100-DMturb stop
```

### 9.3. Monitoring

```bash
cacao-modalstatsTUI
```


## 10. Predictive Control

### 10.1. Pseudo-OL reconstruction

OPTIONAL: Tune software latency and WFS factor to ensure exact pseudoOL reconstruction.

```bash
cacao-fpsctrl setval mfilt auxDMmval.enable ON
cacao-fpsctrl setval mfilt auxDMmval.mixfact 1.0
cacao-fpsctrl setval mfilt auxDMmval.modulate OFF

cacao-fpsctrl setval mfilt loopgain 0.03
cacao-fpsctrl setval mfilt loopmult 0.999

cacao-aorun-080-testOL -w 1.0

# repeat multiple times to converge to correct parameters
cacao-aorun-080-testOL -w 0.1
```

Check that probe and psOL reconstruction overlap and have same amplitude:
```bash
gnuplot
plot [0:] "vispyr2-rundir/testOL.log" u 1:2 w l title "probe", "vispyr2-rundir/testOL.log" u ($1):5 title "psOL", "vispyr2-rundir/testOL.log" u ($1):3 title "DM"
quit
```
The x-offset is the total latency (hardw+softw).



### 10.2. Modal control blocks

Start process mctrlstats to split telemetry into blocks.

```bash
cacao-aorun-120-mstat start
```

Start mkPFXX-Y processes.
```bash
cacao-aorun-130-mkPF 0 start
cacao-aorun-130-mkPF 1 start
```

Start applyPFXX-Y processes.
```bash
cacao-aorun-140-applyPF 0 start
cacao-aorun-140-applyPF 1 start
```


# Cleanup

From main directory (upstream of rootdir) :

```bash
cacao-msglogCTRL stop
cacao-task-manager -C 0 scexao-vispyr-bin2
rm -rf .vispyr2.cacaotaskmanager-log
# Erase everything in shared memory (optional)
rm -rf ${MILK_SHM_DIR}/*
```

# Logging streams to disk





THE END
