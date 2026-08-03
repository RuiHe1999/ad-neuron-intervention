#!/bin/bash -l
#SBATCH --job-name=npsyall
#SBATCH -p high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --array=0-74
#SBATCH --output=logs/npsy_rev_%A_%a.out
#SBATCH --error=logs/npsy_rev_%A_%a.err

source activate graph

roles=(
  "farmer" "welder" "miner" "cleaner" "builder"
  "nurse" "doctor" "truck driver" "airport ground staff" "emergency dispatcher"
  "accountant" "bank clerk" "insurance agent" "civil servant" "salesperson"
  "librarian" "teacher" "scientist" "engineer" "architect"
  "police officer" "soldier" "firefighter" "security guard" "prison officer"
)

conditions=(
  "-0.6Alpha_2000Neurons" "-0.6Alpha_10000Neurons" "-0.6Alpha_AllSigNeurons" 
)

n_roles=${#roles[@]}
n_conds=${#conditions[@]}

tid=${SLURM_ARRAY_TASK_ID}

cond_idx=$(( tid / n_roles ))
role_idx=$(( tid % n_roles ))


role="${roles[$role_idx]}"
condition="${conditions[$cond_idx]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${tid} role=${role} condition=${condition}"

python neuropsy_tests.py --model_type="$condition" --role="$role"

