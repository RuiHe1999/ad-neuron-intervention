#!/bin/bash -l
#SBATCH --job-name=npsyorg
#SBATCH -p high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --output=logs/npsy_org_%A_%a.out
#SBATCH --error=logs/npsy_org_%A_%a.err
#SBATCH --array=0-24

source activate graph
roles=(
  "farmer" "welder" "miner" "cleaner" "builder"
  "nurse" "doctor" "truck driver" "airport ground staff" "emergency dispatcher"
  "accountant" "bank clerk" "insurance agent" "civil servant" "salesperson"
  "librarian" "teacher" "scientist" "engineer" "architect"
  "police officer" "soldier" "firefighter" "security guard" "prison officer"
)
role="${roles[$SLURM_ARRAY_TASK_ID]}"
python neuropsy_tests.py --model_type Original --role "$role"
