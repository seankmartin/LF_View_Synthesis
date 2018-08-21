#!/bin/bash

name=`date --iso-8601=seconds`
outdir="out"
mkdir -p $outdir
sbatch --job-name=$name.vrun --output=$outdir/$name.out slurm-gpu-job.sh "$@"
