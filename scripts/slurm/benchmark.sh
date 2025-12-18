#!/bin/bash
#
#SBATCH --job-name="run_batch"
#SBATCH --account=education-me-msc-me
#SBATCH --partition=gpu-a100
#SBATCH --time=0:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output logs/benchmark%A_%a.out
#SBATCH --error logs/benchmark%A_%a.err

module purge
module load 2025
module load miniconda3/4.12.0

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd /home/rknetemann/projects/oscidyn

conda activate oscidyn

python tests/midterm/benchmarks/symmetry_breaking_resonator/benchmark.py

conda deactivate
