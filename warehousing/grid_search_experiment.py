import numpy as np
import torch
import json

from dp import dynaplex
from psac_env import PSACEnv


def grid_search_experiment(exp_id, mdp_config_file, start_rollout_len, end_rollout_len, rollout_len_step=1):

    folder_name = "collaborative_picking"
    path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, mdp_config_file)

    try:
        with open(path_to_json, "r") as input_file:
            vars = json.load(input_file)  # vars can be initialized manually with something like
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
    except:
        raise Exception(
            "Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")

    mdp = dynaplex.get_mdp(**vars)
    env = PSACEnv(mdp, timestep_delta=vars['timestep_delta'], startup_duration=vars['startup_duration'], max_episode_len=2200)

    for rollout_len in range(start_rollout_len, end_rollout_len, rollout_len_step):

        rewards = []
        for ep in range(10):

            obs, info = env.reset()
            tot_reward = 0

            done = False

            n_act = 0
            while not done:

                action = torch.tensor([[rollout_len]])
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                tot_reward += reward
                n_act += 1

            rewards.append(tot_reward)

        print(f"Exp id: {exp_id}, Rollout len: {rollout_len}")
        print(f"Exp id: {exp_id}, Mean reward: {np.mean(rewards).round(2)} +- {np.std(rewards).round(2)}")
        print()
