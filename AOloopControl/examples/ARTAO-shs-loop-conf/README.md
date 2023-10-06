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

## 2. Start image acquisition and DM
# select hardware mode
./scripts/aorun-setmode-hardw
# Note: A simulation mode is not implemented as this state, but may be added in the future.


The [camera control process](https://github.com/jbOhland93/ximea-xib-64-2-milk) has already started, but the image acquisiton needs to be launched:

```bash
# Attach to the camera control terminal
tmux a -t shsCam-8

# Display control options
help

# Start the acquisition
start

# Set the FPS of the acquisition
setFPS 3000

# Keep this tmux session open for camera control or detach using ctrl+b, d.
```
The [DM control process](https://github.com/jbOhland93/milk-2-dynamic-optics-dm) will automatically forward updates of the ``aol8_dmdisp`` image stream. This stream is created by the dmdisp milk process, which can be launched using a premade cacao script:
```bash
# Start dm comb process
cacao-aorun-000-dm start
```


## 3. Measure SHS camera dark
Block the beam towards the SHS camera, then record the average of the stream.
```bash
# Average aolx_shsCam, store in aolx_shsCam_AVG and save/log as fits file.
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
# Reshapes aolx_shsEval_wfOut to aolx_shsEval_wfOut_reshape
./scripts/aorun-004-reshape-to-wfs-pupil shsWf start
```
Note: Other options besides ``shsWf`` include ``shsGrad`` for the gradient output and ``shsInt`` for the intensity output. However, these outputs are not served by the evaluation process by default as the process of copying data from the GPU into host memory takes time. Therefore, they have to be turned on in the FPS of the shs evaluation process manually (shsOnGPU-Eval-x.comp.cpyGradToCPU and shsOnGPU-Eval-x.comp.cpyIntToCPU). This is not done by the reshaping script as the reshaping is solely for monitoring purpose and shall not change the performance of the loop and the user should turn on these options conciously, if desired.

## 6. Start WF referencing
During operation, changing the zero point offset in WFS space can be useful, e.g. for focal spot optimization or zero-point setting. In this example, we are abusing the wavefront sensor acquisition function that usually processes pyramid WFS data into a format that can be used for AO loop control. As we already have a fully reconstructed WF that can be used as linear input, the dark subtraction and normalization are turned off in the fpssetup.setval.conf file.
```bash
# Start the WFS acquisiton process, which serves as WF manipulator in this example
cacao-aorun-025-acqWFS -w start
```

This process can also be used to set the current WF as reference point. For this, a fixed number of WF samples is collected and then immediately set to be subtracted from each measured WF.
```bash
# Acquire WFS reference
cacao-aorun-026-takeref -n 2000
# Reshape the result for visual inspection
./scripts/aorun-004-reshape-to-wfs-pupil acqWfC start
# Reshape the temporal moving average for visual inspection
# Observe the reshaped image as you switch on and off the WF averaging
./scripts/aorun-004-reshape-to-wfs-pupil acqWfAVG start
```

## 5. Measure DM to WFS latency

```bash
# Measure latency
cacao-aorun-020-mlat -w
```



## 6. Acquire Calibration


### 6.1. Prepare DM poke modes

```bash
# Create DM poke mode cubes
cacao-mkDMpokemodes -z 5 -c 25
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



### 6.2. Run acquisition


```bash
# Acquire response matrix - Hadamard modes
# 4 cycles - default is 10.
cacao-aorun-030-acqlinResp -n 4 HpokeC
```
This could take a while. Check status on milk-procCTRL.
To inspect results, display file conf/RMmodesWFS/HpokeC.WFSresp.fits.

### Decode Hadamard matrix

```bash
cacao-aorun-031-RMHdecode
```
To inspect results, display file conf/RMmodesWFS/zrespM-H.fits.
This should visually look like a zonal response matrix.


### 6.3. Make DM and WFS masks

```bash
cacao-aorun-032-RMmkmask
```
Check results:
- conf/dmmask.fits
- conf/wfsmask.fits

If needed, rerun command with non-default parameters (see -h for options).
Note: we are not going to apply the masks in this example, so OK if not net properly. The masks are informative here, allowing us to view which DM actuators and WFS pixels have the best response.


### 6.4. Create synthetic (Fourier) response matrix

```bash
cacao-aorun-033-RM-mksynthetic -c 25 -a 2.0
```


## 7. Compute control matrix (straight)

Compute control modes, in both WFS and DM spaces.
Set GPU device (if GPU available).

```bash
cacao-fpsctrl setval compstrCM svdlim 0.002
cacao-fpsctrl setval compstrCM GPUdevice 0
```
Then run the compstrCM process to compute CM and load it to shared memory :
```bash
cacao-aorun-039-compstrCM
```

Check results:
- conf/CMmodesDM/CMmodesDM.fits
- conf/CMmodesWFS/CMmodesWFS.fits


## 8. Running the loop

### 8.1. Core processes

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


python -m pycacao.calib.mkmodes randhad HpokeCrand.fits (edited) 
python -m pycacao.calib.rmdecode


Closing the loop and setting loop parameters with mfilt:

```bash
# Set loop gain
cacao-fpsctrl setval mfilt loopgain 0.1

# set modal gains, mults and limits
cacao-aorun-061-setmgains 0.8 -f 0.05 -t 1.2
cacao-aorun-062-setmmults 0.05 -f 0.9 -t 1.0
cacao-aorun-063-setmlimits 0.8 -f 0.05 -t 1.0

# Set loop mult
cacao-fpsctrl setval mfilt loopmult 0.98

# close loop
cacao-fpsctrl setval mfilt loopON ON

```

Misc tools:

```bash
# Astrogrid control
cacao-DMastrogrid start
cacao-DMastrogrid stop

# Set WFS reference to flat illumination over wfsmask
cacao-wfsref-setflat
```

scexao-specific tools
```bash

```



### 8.2. Zero Point Offsetting

```bash
cacao-aorun-071-zpo start
```

Select DM channels to be included in zpo.


## 9. Testing the loop

### 9.1. SelfRM

```bash
# Set max number of modes above nbmodes to measure all modes

cacao-fpsctrl runstop mfilt 0 0
cacao-fpsctrl setval mfilt selfRM.zsize 20
cacao-fpsctrl setval mfilt selfRM.NBmode 2000
cacao-fpsctrl runstart mfilt 0 0
cacao-fpsctrl setval mfilt selfRM.enable ON
```

Check result: vispyr2-rundir/selfRM.fits

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
```



# Logging streams to disk





THE END
