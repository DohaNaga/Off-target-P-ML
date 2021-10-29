#!/bin/bash
#SBATCH --job-name "offtargets_tuning"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 24:0:0


ml purge
ml R/3.5.1-goolf-1.7.20
Rscript "/tuning_1.R"