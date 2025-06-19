#!/bin/bash
# commands to run a game experiment on Snellius, the Dutch academic hpc server.
#SBATCH -p rome -t 24:00:00 --ntasks-per-node 1 --ntasks 1 --cpus-per-task 16
eval "$(conda shell.bash hook)"
conda activate psac
srun python -m games.experiment_main --experiment=connect_four_configuration_exp_c