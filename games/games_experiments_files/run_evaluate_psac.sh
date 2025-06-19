#!/bin/bash
# commands to run a game experiment on Snellius, the Dutch academic hpc server.
#SBATCH -p rome -t 4:00:00 --ntasks-per-node 1 --ntasks 1 --cpus-per-task 16
eval "$(conda shell.bash hook)"
conda activate psac
experiment="connect_four_configuration_exp_c"
python -m games.model_performances.model_configuration_and_baseline_vs_random_c --experiment=$experiment --nr_games=500 --nr_processes=16
