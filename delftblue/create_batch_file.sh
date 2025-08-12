#!/bin/bash
#
#SBATCH --job-name="create_batch_file"
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-me-msc-me

cd /home/rknetemann/projects/oscidyn

source .venv/bin/activate

srun python tests/batching/create_batch_file.py tmp/batch_file.hdf5

deactivate

