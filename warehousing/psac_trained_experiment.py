import torch
import json
import tianshou as ts

from torch.optim.lr_scheduler import ExponentialLR

from dp import dynaplex
from networks.actor_critic_mlp_per_node import NodeMLPActor4Time, NodeMLPCritic4Time
from psac_env import PSACEnv


def psac_trained_experiment(mdp_config_file, network_file, n_ep_start, n_ep_end):
    # Training parameters
    train_args = {"hidden_dim": 64,
                  "lr": 5e-4,
                  "discount_factor": 0.99,
                  "batch_size": 64,
                  "max_batch_size": 0,  # 0 means step_per_collect amount
                  "nr_train_envs": 1,
                  "nr_test_envs": 1,
                  "max_epoch": 25,
                  "step_per_collect": 10000,
                  "step_per_epoch": 20000,
                  "repeat_per_collect": 2,
                  "replay_buffer_size": 20000,
                  "max_batchsize": 2048,
                  # train environments can be either infinite or finite horizon mdp. 0 means infinite horizon
                  "num_periods_until_done": 2200,
                  # in order to use test environments, episodes should be guaranteed to get to terminations
                  "num_steps_per_test_episode": 2200
                  }

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features_per_node = 7

    # define actor network structure
    actor_net = NodeMLPActor4Time(
        input_dim=n_features_per_node,
        hidden_dim=train_args["hidden_dim"],
        output_dims=[20],
        n_nodes=351,
        min_val=torch.finfo(torch.float).min
    ).to(device)

    # define critic network structure
    critic_net = NodeMLPCritic4Time(
        input_dim=n_features_per_node,
        hidden_dim=train_args["hidden_dim"],
        n_nodes=351
    ).to(device).share_memory()

    # define optimizer
    optim = torch.optim.Adam(
        params=list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=train_args["lr"]
    )

    # define scheduler
    scheduler = ExponentialLR(optim, 0.99)

    # define PPO policy
    policy = ts.policy.PPOPolicy(actor_net, critic_net, optim,
                                 discount_factor=train_args["discount_factor"],
                                 max_batchsize=train_args["max_batchsize"],     # max batch size for GAE estimation, default to 256
                                 value_clip=True,
                                 dist_fn=torch.distributions.categorical.Categorical,
                                 deterministic_eval=True,
                                 lr_scheduler=scheduler,
                                 reward_normalization=False
                                 )
    policy.action_type = "discrete"

    policy_state = network_file
    policy.load_state_dict(
        torch.load(policy_state)
    )
    policy.eval()

    env = PSACEnv(mdp, timestep_delta=vars['timestep_delta'], startup_duration=vars['startup_duration'],  max_episode_len=2200)    # timestep 10, 30 min

    rewards = []
    for ep in range(n_ep_start, n_ep_end):

        obs, info = env.reset(ep, vars['startup_duration'])

        tot_reward = 0

        done = False

        while not done:

            obs = ts.data.Batch(ts.data.Batch({'obs': [obs], 'info': {}}))
            action = policy.forward(obs).act

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            tot_reward += reward

        rewards.append(tot_reward)

    return rewards
