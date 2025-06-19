import torch
import json
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from torch.optim.lr_scheduler import ExponentialLR

from dp import dynaplex

from networks.actor_critic_mlp_per_node import NodeMLPActor4Time, NodeMLPCritic4Time

from psac_env import PSACEnv

folder_name = "collaborative_picking"
path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, f"mdp_config_non_stat_4.json")

try:
    with open(path_to_json, "r") as input_file:
        vars = json.load(input_file)  # vars can be initialized manually with something like
except FileNotFoundError:
    raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
except:
    raise Exception(
        "Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")

mdp = dynaplex.get_mdp(**vars)

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


def save_best_fn(policy):
    print("Saving improved policy")
    # torch.save(policy.state_dict(), f"trained_policies/psac_non_stat_coll_pick.pt")
    torch.save(policy.state_dict(), f"trained_policies/test.pt")


def get_env():
    return PSACEnv(mdp, timestep_delta=vars['timestep_delta'], startup_duration=vars['startup_duration'], max_episode_len=train_args["num_periods_until_done"])


def get_test_env():
    return PSACEnv(mdp, timestep_delta=vars['timestep_delta'], startup_duration=vars['startup_duration'], max_episode_len=train_args["num_steps_per_test_episode"])


def preprocess_function(**kwargs):
    """
    Observations contain the mask as part of a dictionary.
    This function ensures that the data gathered in training and testing are in the correct format.
    """
    if "obs" in kwargs:
        obs_with_tensors = [
            {"obs": obs['obs'].to(device=device),
             "mask": obs['mask'].to(device=device)}
            for obs in kwargs["obs"]]
        kwargs["obs"] = obs_with_tensors
    if "obs_next" in kwargs:
        obs_with_tensors = [
            {"obs": obs['obs'].to(device=device),
             "mask": obs['mask'].to(device=device)}
            for obs in kwargs["obs_next"]]
        kwargs["obs_next"] = obs_with_tensors
    return kwargs


if __name__ == '__main__':

    n_features_per_node = 7

    model_name = "ppo_model_dict.pt"     #used for tensorboard logging
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
                                 max_batchsize=train_args["max_batchsize"], # max batch size for GAE estimation, default to 256
                                 value_clip=True,
                                 dist_fn=torch.distributions.categorical.Categorical,
                                 deterministic_eval=True,
                                 lr_scheduler=scheduler,
                                 reward_normalization=False
                                 )
    policy.action_type = "discrete"

    # a tensorboard logger is available to monitor training results.
    # log in the directory where all mdp results are stored:
    log_path = dynaplex.filepath(mdp.identifier(), "tensorboard_logs", model_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # create nr_envs train environments
    # note that actual environment parallelization does not work currently because of limitations
    # in the C++ based simulation framework
    train_envs = ts.env.DummyVectorEnv(
        [get_env for _ in range(train_args["nr_train_envs"])]
    )
    collector = ts.data.Collector(policy, train_envs,
                                  ts.data.VectorReplayBuffer(train_args["replay_buffer_size"],
                                                             train_args["nr_train_envs"]),
                                  preprocess_fn=preprocess_function)
    collector.reset()

    # create nr_envs test environments
    test_envs = ts.env.DummyVectorEnv(
        [get_test_env for _ in range(train_args["nr_test_envs"])]
    )
    test_collector = ts.data.Collector(policy, test_envs, preprocess_fn=preprocess_function)
    test_collector.reset()

    # train the policy
    print("Starting training")
    policy.train()
    trainer = ts.trainer.OnpolicyTrainer(
        policy, collector, test_collector=test_collector,
        # policy, collector, test_collector=None,
        max_epoch=train_args["max_epoch"],
        step_per_epoch=train_args["step_per_epoch"],
        step_per_collect=train_args["step_per_collect"],
        episode_per_test=5, batch_size=train_args["batch_size"],
        repeat_per_collect=train_args["repeat_per_collect"],
        logger=logger, test_in_train=True,
        save_best_fn=save_best_fn
    )
    result = trainer.run()
    print(f'Finished training!')
