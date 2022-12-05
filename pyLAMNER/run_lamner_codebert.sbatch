#!/bin/bash

#SBATCH --job-name=lamner_codebert
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=8:00:00

## User python environment
export HOME=/data/healthy-ml/scratch/mingy
eval "$(/data/healthy-ml/scratch/mingy/anaconda3/bin/conda shell.bash hook)"
source activate $HOME/anaconda3/envs/pylamner
export PYTHON=$HOME/anaconda3/envs/pylamner/bin/python
export HOME=/data/healthy-ml/scratch/mingy/pyLAMNER/pyLAMNER

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' | sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= " $SLURM_NTASKS_PER_NODE

#### Setup and run job
echo # Run started at:- "
date

# Run
srun python codebert_baseline_clean.py

echo "Run completed at:- "
date
