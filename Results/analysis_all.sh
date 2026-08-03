#!/bin/bash -l 
#SBATCH --job-name=analy 
#SBATCH -p medium 
#SBATCH -N 1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=40G 
#SBATCH --output=analysis_%j.out 
#SBATCH --error=analysis_%j.err 

source activate graph 

python analysis_immediate.py
python analysis_delay.py
python analysis_verbal_fluency.py
python analysis_working_memory.py
python analysis_procedure.py
python analysis_scene.py
python analysis_coreference.py
python analysis_summary.py