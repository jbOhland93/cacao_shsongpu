# Overview

SCExAO system pyramid WFS.
Low-resolution WFS mode (120x120)

cacao-task-manager tasks for this example :

~~~
 0           INITSETUP             DONE        READY   Initial setup:
 1     GETSIMCONFFILES             DONE        READY   Get simulation files:
 2          TESTCONFIG             DONE        READY   Test configuration:
 3          CACAOSETUP             DONE        READY   Run cacao-setup:
~~~
Subsequent tasks can perform specific parts of the AO loop.




# Running the example

:warning: Check the [instructions](https://github.com/cacao-org/cacao/tree/dev/AOloopControl/examples) before running these steps

## Setting up processes

```bash
# Deploy configuration :
# download from source and start conf processes
cacao-loop-deploy scexao-NIRPL

# Go to rootdir, from which user controls the loop
cd NIRPL-rootdir

# select simulation mode
./scripts/aorun-setmode-hardw
# (alternatively, run ./scripts/aorun-setmode-sim to connect to simulator)
```


## Measure WFS dark


```bash
cacao-aorun-005-takedark
```



## Start WFS acquisition

```bash
# Acquire WFS frames
cacao-aorun-025-acqWFS start
```


## Measure DM to WFS latency

```bash
# Measure latency
cacao-aorun-020-mlat -w
```



## Acquire response matrix


### Prepare DM poke modes

```bash
# Create DM poke mode cubes
cacao-mkDMpokemodes
```
The following files are written to ./conf/DMmodes/ :
- DMmask.fits    : DM mask
- Fmodes.fits    : Fourier modes
- Zmodes.fits    : Zernike modes
- HpokeC.fits    : Hadamard modes
- Hmat.fits      : Hadamard matrix (to convert Hadamard-zonal)
- Hpixindex.fits : Hadamard pixel index



### Run acquisition


```bash
# Acquire response matrix - Fourier modes
cacao-fpsctrl setval measlinresp procinfo.loopcntMax 20
cacao-aorun-030-acqlinResp Fmodes

# NOTE: Alternate option is Hadamard modes
# Acquire response matrix - Hadamard modes
#cacao-fpsctrl setval measlinresp procinfo.loopcntMax 3
#cacao-aorun-030-acqlinResp HpokeC
```

### Take reference

```bash
# Acquire reference
cacao-aorun-026-takeref
```


## Compute control matrix (straight)

Compute control modes, in both WFS and DM spaces.

```bash
cacao-fpsctrl setval compstrCM RMmodesDM "../conf/RMmodesDM/Fmodes.fits"
cacao-fpsctrl setval compstrCM RMmodesWFS "../conf/RMmodesWFS/Fmodes.WFSresp.fits"
cacao-fpsctrl setval compstrCM svdlim 0.2
```
Then run the compstrCM process to compute CM and load it to shared memory :
```bash
cacao-aorun-039-compstrCM
```



## Running the loop

Select GPUs
```bash
cacao-fpsctrl setval wfs2cmodeval GPUindex 0
cacao-fpsctrl setval mvalC2dm GPUindex 3
```


From directory vispyr-rootdir, start 3 processes :

```bash
# start WFS -> mode coefficient values
cacao-aorun-050-wfs2cmval start

# start modal filtering
cacao-aorun-060-mfilt start

# start mode coeff values -> DM
cacao-aorun-070-cmval2dm start

```

Closing the loop and setting loop parameters with mfilt:

```bash
# Set loop gain
cacao-fpsctrl setval mfilt loopgain 0.1

# Set loop mult
cacao-fpsctrl setval mfilt loopmult 0.98

# close loop
cacao-fpsctrl setval mfilt loopON ON

```


THE END
