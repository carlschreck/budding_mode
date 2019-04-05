#!/bin/bash

# Set directories
outdir=../data/
rundir=../code/

# geometric parameters
ar1=1.01
ar2=2.0

# rates
rate0=1d0
b=4e3

# steps
steps=1250000
layerskip=800
dataskip=50000
prodskip=1250000
restskip=50000
dt=5e-6

# growth layer widths
layerwidth=2.1
layerdepth=4.0
propdepth=100.0
bounddepth=0.0

# logical variables
movie=.true.
restart=.true.

# run parameters
desync=0.4

# division mode
divmode=4

cd $outdir

for seed in `seq -1820 -1001`
do

# output files
suffix=radial_layer${layerdepth}_desync${desync}_b${b}_seed${seed}_divmode${divmode}.dat
prodfile=prod_$suffix
restfile=restart_$suffix
radfile=radius_$suffix

# run program
time $rundir/dimers_damped_radial_removecells_buddingmode.o <<EOF
  $ar1
  $ar2
  $rate0
  $b
  $steps
  $layerskip
  $dataskip
  $prodskip
  $restskip
  $dt
  $layerwidth
  $layerdepth
  $propdepth
  $bounddepth
  $desync
  $seed
  $prodfile 
  $restfile 
  $radfile 
  $divmode
  $movie
  $restart
EOF

done
