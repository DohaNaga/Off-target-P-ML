#!/bin/bash
#SBATCH --job-name "excape_casestudies"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 18
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition gpu
#SBATCH --gres gpu:3
#SBATCH --time 48:0:0


ml purge
ml R/3.5.1-goolf-1.7.20
Rscript "tuning_1.R"
