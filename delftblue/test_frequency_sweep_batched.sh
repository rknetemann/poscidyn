#!/bin/bash
#
#SBATCH --job-name="test_frequency_sweep_batched"
#SBATCH --partition=gpu-a100
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8192M
#SBATCH --account=education-me-msc-me

echo "Starting test_frequency_sweep_batched.sh"

# Necessary for GPU utilization information
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/frequency_sweep/test_frequency_sweep_batched.py

deactivate

# Necessary for GPU utilization information
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
