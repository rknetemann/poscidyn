#!/bin/bash
#
#SBATCH --job-name="run_batch"
#SBATCH --partition=gpu-a100
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-me-msc-me
#SBATCH --array=0-99

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/batching/batch.py $SLURM_ARRAY_TASK_ID

deactivate

