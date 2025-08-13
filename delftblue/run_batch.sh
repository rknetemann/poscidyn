#!/bin/bash
#
#SBATCH --job-name="run_batch"
#SBATCH --account=education-me-msc-me
#SBATCH --partition=gpu-a100
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output output/slurm%A-%a.out
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-99

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/batching/batch.py --n_batches 100 --batch_id $SLURM_ARRAY_TASK_ID --n_parallel_sim 128 --file_name "output/batch_$(date +%Y-%m-%d_%H:%M:%S)_$SLURM_ARRAY_TASK_ID.hdf5"

deactivate

