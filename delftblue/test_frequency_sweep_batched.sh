#!/bin/bash
#
#SBATCH --job-name="test_frequency_sweep_batched"
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-me-msc-me

cd /home/rknetemann/projects/oscidyn
source .venv/bin/activate

# Necessary for GPU utilization information
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python tests/frequency_sweep/test_frequency_sweep_batched.py

# Necessary for GPU utilization information
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

deactivate