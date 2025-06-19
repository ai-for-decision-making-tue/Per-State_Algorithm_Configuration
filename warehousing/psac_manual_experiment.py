import json

import torch
from dp import dynaplex
from psac_env import PSACEnv


def psac_manual_experiment(mdp_config_file, n_ep_start, n_ep_end):

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

    env = PSACEnv(mdp, timestep_delta=vars['timestep_delta'], startup_duration=vars['startup_duration'],  max_episode_len=2200)    # timestep 10, 30 min

    rewards = []
    for ep in range(n_ep_start, n_ep_end):

        obs, info = env.reset(ep)

        tot_reward = 0

        done = False

        while not done:

            dist_type = obs['obs'][1:5]
            if dist_type[0] == 1:
                # rollout len: 25
                action = torch.tensor([[4]])
            elif dist_type[1] == 1:
                # rollout len: 30
                action = torch.tensor([[5]])
            elif dist_type[2] == 1:
                # rollout len: 50
                action = torch.tensor([[9]])
            else:
                # rollout len: 85
                action = torch.tensor([[16]])

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            tot_reward += reward

        rewards.append(tot_reward)

    return rewards
