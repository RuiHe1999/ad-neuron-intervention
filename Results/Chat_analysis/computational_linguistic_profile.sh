#!/bin/bash -l 
#SBATCH --job-name=cl 
#SBATCH -p high 
#SBATCH -N 1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=40G 
#SBATCH --array=0-2 
#SBATCH --output=logs/cl_profile_%A_%a.out 
#SBATCH --error=logs/cl_profile_%A_%a.err 

mkdir -p logs 

source activate graph 

tasks=( "scene" "immediate_recall" "delayed_recall" ) 
tid=${SLURM_ARRAY_TASK_ID} 
task_name="${tasks[$tid]}" 

echo "SLURM_JOB_ID=${SLURM_JOB_ID}" 
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}" 
echo "SLURM_ARRAY_TASK_ID=${tid}" 
echo "task_name=${task_name}" 
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}" 

nvidia-smi || true 

python computational_linguistic_profile.py "${task_name}"