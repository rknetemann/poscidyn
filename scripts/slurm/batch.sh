#!/bin/bash
#
#SBATCH --job-name="run_batch"
#SBATCH --account=education-me-msc-me
#SBATCH --partition=gpu-a100
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output logs/coupled_2_dof%A_%a.out
#SBATCH --error logs/coupled_2_dof%A_%a.err
#SBATCH --array=0-1

module purge
module load 2025
module load miniconda3/4.12.0

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99

cd /home/rknetemann/projects/oscidyn

conda activate oscidyn

OUTDIR="results/raw"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

python scripts/batch_simulation/coupled_2_dof.py --n_tasks=2 --task_id=$SLURM_ARRAY_TASK_ID --n_parallel_sim=64 --file_name="$OUTDIR/batch_${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}.hdf5"

conda deactivate
