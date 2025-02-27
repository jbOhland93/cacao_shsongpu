#!/usr/bin/env bash
# This file will be sourced by cacao-setup and high-level cacao scripts


export CACAO_LOOPNAME="maps"
export CACAO_LOOPNUMBER="2"

# ====== DEFORMABLE MIRROR ==========

# Deformable mirror (DM) size
# If DM is single dimension, enter "1" for DMsize
#
export CACAO_DMINDEX="00"    # Hardware DM - connected to physical DM
export CACAO_DMSIMINDEX="10" # Simulation DM
export CACAO_DMxsize="336"
export CACAO_DMysize="1"


# 1 if DM actuators are on a coordinate grid
# This informs processes if a spatial relationship exists
# between DM actuators
export CACAO_DMSPATIAL="0"



# ====== DIRECTORIES ================

# Root directory
export CACAO_LOOPROOTDIR="${CACAO_LOOPNAME}-rootdir"

# Run directory. This is a subdirectory of rootdir
# processes run in CACAO_LOOPROOTDIR/CACAO_LOOPRUNDIR
export CACAO_LOOPRUNDIR="${CACAO_LOOPNAME}-rundir"

# input WFS stream
export CACAO_WFSSTREAM="vispyrcam"    # Hardware stream, connected to physical camera
export CACAO_WFSSTREAMSIM="mapsvcamsim"

# Specify that WFS stream is not raw image, but processed WFS signal
# This turns off intensity scaling
export CACAO_WFSSTREAM_PROCESSED="ON"

export CACAO_LOOPDATALOGDIR="$(pwd)/datalogdir"


# ========================================
#       FPS processes to be set up
# ========================================

# DM combination
# Manages mutipe DM channels
#
export CACAO_FPSPROC_DMCH2DISP="ON"
export CACAO_FPSPROC_DMCH2DISPSIM="ON"

# DM turbulence simulator
export CACAO_FPSPROC_DMATMTURB="ON"

# Delay stream: emulates time lag in hardware
# Used to simulate a time lag
#
export CACAO_FPSPROC_DMSIMDELAY="ON"

# MVM lop on GPU: used to simulate hardware
#
export CACAO_FPSPROC_SIMMVMGPU="ON"


# Measure hardware latency
#
export CACAO_FPSPROC_MLAT="ON"

# Acquire WFS stream
#
export CACAO_FPSPROC_ACQUWFS="ON"

# Acquire linear RM
#
export CACAO_FPSPROC_MEASURELINRESP="ON"

# Compute control matrix - straight
#
export CACAO_FPSPROC_COMPSTRCM="ON"

# Extract control modes
#
export CACAO_FPSPROC_MVMGPU_WFS2CMODEVAL="ON"

# Modal control filtering
#
export CACAO_FPSPROC_MODALFILTERING="ON"

# Compute DM command from control mode values
#
export CACAO_FPSPROC_MVMGPU_CMODEVAL2DM="ON"
