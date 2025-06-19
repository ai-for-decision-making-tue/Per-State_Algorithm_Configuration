#!/bin/bash
# commands to run a game experiment on Snellius, the Dutch academic hpc server.
#SBATCH -p rome -t 36:00:00 --ntasks-per-node 1 --ntasks 1 --cpus-per-task 16
eval "$(conda shell.bash hook)"
conda create -n game_wrapper_openspiel python=3.11 --y
conda activate game_wrapper_openspiel
python -m pip install open_spiel
python -m pip install numpy==1.24.0 tensorflow==2.15.0 tensorflow-probability==0.22.1 tensorflow_datasets==4.9.7 keras==2.15.0
PYTHONPATH=$(pwd) python games/train_open_spiel_player_nr/run_alpha_zero.py --game connect_four --learning_rate 0.001 --max_steps 1000 --max_simulations 100 --nn_width 64 --nn_depth 3 --path models/connect_four_long_run --eval_levels 7 --actors 10 --evaluators 4 --checkpoint_freq 5