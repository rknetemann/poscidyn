#!/bin/bash
#
#SBATCH --job-name="test_frequency_sweep_batched"
#SBATCH --partition=gpu-a100
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-me-msc-me

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export N_SIMULATIONS_IN_PARALLEL_PER_GPU=100
export N_DUFFING=4000

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/test_simulation_batch.py

deactivate

