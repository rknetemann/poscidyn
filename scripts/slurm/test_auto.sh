#!/bin/bash
#
#SBATCH --job-name="test_auto"
#SBATCH --account=education-me-msc-me
#SBATCH --partition=compute-p2
#SBATCH --time=0:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=500M
#SBATCH --output logs_auto/test_auto%A_%a.out
#SBATCH --error logs_auto/test_auto%A_%a.err

module purge
module load 2025
module load python
module load py-numpy
module load py-matplotlib

source /home/rknetemann/auto/07p/cmds/auto.env.sh

cd auto/simulations/symmetry_breaking/

auto frc.autoP
