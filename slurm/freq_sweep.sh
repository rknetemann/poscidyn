#!/bin/bash
#
#SBATCH --job-name="run_freq_sweep"
#SBATCH --account=education-me-msc-me
#SBATCH --partition=gpu-a100
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output logs/freq_sweep%A_%a.out
#SBATCH --error logs/freq_sweep%A_%a.err

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/frequency_sweep/test_frequency_sweep.py

deactivate