#!/bin/bash

#SBATCH -J codet5
#SBATCH -o codet5_1e-2_20epochs%j.out
#SBATCH -e codet5_1e-2_20epochs%j.err
#SBATCH --mail-user=ming1022@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:2
#SBATCH --mem 64GB
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --open-mode append
#SBATCH --partition healthyml
#SBATCH --time=24:00:00

## User python environment
export HOME=/data/healthy-ml/scratch/mingy
eval "$(/data/healthy-ml/scratch/mingy/anaconda3/bin/conda shell.bash hook)"
conda activate $HOME/anaconda3/envs/pylamner
export PYTHON=$HOME/anaconda3/envs/pylamner/bin/python

export CONFIG_DIR=/data/healthy-ml/scratch/mingy/pyLAMNER/pyLAMNER
cd $CONFIG_DIR

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
srun python codet5_baseline.py

echo "Run completed at:- "
date
