import torch
from datetime import datetime
import tianshou as ts
from tianshou.env.venvs import SubprocVectorEnv
import os
import numpy as np
from pathlib import Path
import json
import gymnasium as gym

from torch.utils.tensorboard.writer import SummaryWriter

from games.custom_tensorboard_logger import CustomTensorboardLogger

from torch.optim.lr_scheduler import ExponentialLR
from games.experiments import ExperimentParams
from games.networks.psac_actor_critic_mlp import ResNet
from functools import partial
from games.utils import parse_experiment_params


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is {DEVICE}")


# Training parameters
TRAIN_ARGS = {
    "hidden_dim": 64,
    "lr": 1e-3,
    "eps_clip": 0.2,
    "discount_factor": 1.0,
    "batch_size": 64,
    "max_batch_size": 0,
    "nr_train_envs": 16,
    "nr_test_envs": 16,
    "max_epoch": 20,
    "resnet_nr_resnet_blocks": 2,
    "resnet_filters": 64,
    "resnet_fc_units": 64,
    "step_per_collect": 2_000,
    "step_per_epoch": 40_000,
    "repeat_per_collect": 2,
    "train_replay_buffer_size": 40_000,
    "test_replay_buffer_size": 5_000,
    "max_batchsize": 2_048,
    "num_periods_until_done": 0,
    "num_steps_per_test_episode": 0,
    "num_test_episodes": 600,
}


def preprocess_function(**kwargs):
    """
    Observations contain the mask as part of a dictionary.
    This function ensures that the data gathered in training and testing are in the correct format.
    """
    if "obs" in kwargs:
        obs_with_tensors = _convert_np_array_in_obs_to_torch(kwargs["obs"])
        kwargs["obs"] = obs_with_tensors
    if "obs_next" in kwargs:
        obs_with_tensors = _convert_np_array_in_obs_to_torch(kwargs["obs_next"])
        kwargs["obs_next"] = obs_with_tensors

    return kwargs


def _convert_np_array_in_obs_to_torch(obs_dict: dict) -> dict:
    return {key: _convert_nested_np_arrays_to_torch(obs_dict, key) for key in ["obs", "mask"]}


def _convert_nested_np_arrays_to_torch(obs: np.ndarray, key: str) -> torch.Tensor:
    np_arrays = np.array([i[key] for i in obs], dtype=np.float32)
    return torch.from_numpy(np_arrays).to(DEVICE)


def _get_logger(dir_: Path, file_prefix: str) -> CustomTensorboardLogger:
    file_prefix = f"{file_prefix}.pt"
    log_path = dir_ / file_prefix
    writer = SummaryWriter(log_path)
    logger = CustomTensorboardLogger(writer)
    return logger


def save_best_fn(policy: ts.policy.PPOPolicy, dir_: Path, file_prefix: str):
    print("Saving improved policy")
    model_save_path = dir_ / f"{file_prefix}_ppo.pt"
    torch.save(policy, model_save_path)
    state_dict_save_path = dir_ / f"{file_prefix}_ppo_state_dict.pt"
    torch.save(policy.state_dict(), state_dict_save_path)


def _save_result(env_info: dict, result: ts.data.InfoStats, train_args: dict, dir_: Path, file_prefix: str) -> None:
    result_save_path = dir_ / f"{file_prefix}_results.txt"
    with open(result_save_path, "w") as file:
        file.write("env info:\n")
        file.write(json.dumps(env_info, indent=4))

        file.write("\n\nResults:")

        for attribute in [
            "gradient_step",
            "best_reward",
            "best_reward_std",
            "train_step",
            "train_episode",
            "test_step",
            "test_episode",
        ]:
            value = getattr(result, attribute)
            line = f"{attribute}: {value}\n"
            file.write(line)

        timing_stats = result.timing

        for attribute in [
            "test_time",
            "total_time",
            "train_time",
            "train_time_collect",
            "train_time_update",
            "update_speed",
        ]:
            value = getattr(timing_stats, attribute)
            line = f"{attribute}: {value}\n"
            file.write(line)

        file.write("\ntrain arguments:\n")
        file.write(json.dumps(train_args, indent=4))


def _init_dirs(experiment_params: ExperimentParams) -> tuple[Path, Path]:
    results_dir = experiment_params.results_dir_path
    results_dir.mkdir(exist_ok=True, parents=True)

    tensorboard_log_dir = experiment_params.tensorboard_dir_path
    tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

    return results_dir, tensorboard_log_dir


class NonResettingOnpolicyTrainer(ts.trainer.OnpolicyTrainer):
    def __iter__(self):
        if self.epoch == 0:
            self.reset()
        return self

    def run(self) -> ts.data.InfoStats:
        info = super().run()
        self.epoch -= 1
        self.iter_num -= 1
        return info


def _get_ppo(input_shape, action_space) -> ts.policy.PPOPolicy:
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        num_output_features = sum(action_space.nvec)
        output_f = torch.nn.Softmax
    elif isinstance(action_space, gym.spaces.MultiBinary):
        num_output_features = action_space.n
        output_f = torch.nn.Sigmoid
    elif isinstance(action_space, gym.spaces.Box):
        raise ValueError(f"Unsupported action space: {action_space}")
        # PPO Seems to require logits as output so ReLU is not suitable

        # num_output_features = sum(action_space.shape)
        # # torch.nn.Identity can also be considered
        # output_f = torch.nn.ReLU
    else:
        raise ValueError(f"Unsupported action space: {action_space}")

    actor_net = (
        ResNet(
            input_shape=input_shape,
            num_output_features=num_output_features,
            num_res_block=TRAIN_ARGS["resnet_nr_resnet_blocks"],
            num_filters=TRAIN_ARGS["resnet_filters"],
            num_fc_units=TRAIN_ARGS["resnet_fc_units"],
            output=output_f,
            forward_returns_state=True,
        )
        .to(DEVICE)
        .share_memory()
    )

    # define critic network structure
    critic_net = (
        ResNet(
            input_shape=input_shape,
            num_output_features=1,
            num_res_block=TRAIN_ARGS["resnet_nr_resnet_blocks"],
            num_filters=TRAIN_ARGS["resnet_filters"],
            num_fc_units=TRAIN_ARGS["resnet_fc_units"],
            output=torch.nn.Tanh,
            forward_returns_state=False,
        )
        .to(DEVICE)
        .share_memory()
    )

    optim = torch.optim.Adam(params=list(actor_net.parameters()) + list(critic_net.parameters()), lr=TRAIN_ARGS["lr"])
    scheduler = ExponentialLR(optim, 0.99)

    policy = ts.policy.PPOPolicy(
        actor=actor_net,
        critic=critic_net,
        optim=optim,
        action_space=action_space,
        action_scaling=False,
        discount_factor=TRAIN_ARGS["discount_factor"],
        max_batchsize=TRAIN_ARGS["max_batchsize"],  # max batch size for GAE estimation, default to 256
        value_clip=True,
        eps_clip=TRAIN_ARGS["eps_clip"],
        dist_fn=torch.distributions.categorical.Categorical,
        deterministic_eval=True,
        lr_scheduler=scheduler,
        reward_normalization=False,
    )
    policy.action_type = "discrete"
    return policy


def _get_sub_proc_vector_env(nr_envs: int, exp_params: ExperimentParams) -> SubprocVectorEnv:
    core_nrs_available = list(os.sched_getaffinity(0))
    assert nr_envs <= len(core_nrs_available), f"Nr of envs {nr_envs} exceeds nr of cores {len(core_nrs_available)}"

    env_fns = [partial(exp_params.create_env_f, core_nr=core_nrs_available[env_id]) for env_id in range(nr_envs)]
    vector_env = SubprocVectorEnv(env_fns)
    return vector_env


if __name__ == "__main__":
    experiment_params, epoch = parse_experiment_params()

    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    results_dir, tensorboard_log_dir = _init_dirs(experiment_params=experiment_params)

    train_envs = _get_sub_proc_vector_env(TRAIN_ARGS["nr_train_envs"], exp_params=experiment_params)
    test_envs = _get_sub_proc_vector_env(TRAIN_ARGS["nr_test_envs"], exp_params=experiment_params)

    input_shape = train_envs.observation_space[0]["obs"].shape
    action_space = train_envs.action_space[0]
    policy = _get_ppo(input_shape=input_shape, action_space=action_space)

    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(TRAIN_ARGS["train_replay_buffer_size"], TRAIN_ARGS["nr_train_envs"]),
        exploration_noise=True,
        preprocess_fn=preprocess_function,
    )
    train_collector.reset()

    test_collector = ts.data.Collector(
        policy,
        test_envs,
        ts.data.VectorReplayBuffer(TRAIN_ARGS["test_replay_buffer_size"], TRAIN_ARGS["nr_test_envs"]),
        exploration_noise=False,
        preprocess_fn=preprocess_function,
    )
    test_collector.reset()

    # train the policy
    logger = _get_logger(tensorboard_log_dir, start_time)
    print("Starting training")
    policy.train()
    trainer = NonResettingOnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1,
        step_per_epoch=TRAIN_ARGS["step_per_epoch"],
        step_per_collect=TRAIN_ARGS["step_per_collect"],
        episode_per_test=TRAIN_ARGS["num_test_episodes"],
        batch_size=TRAIN_ARGS["batch_size"],
        repeat_per_collect=TRAIN_ARGS["repeat_per_collect"],
        logger=logger,
        test_in_train=True,
        save_best_fn=partial(save_best_fn, dir_=results_dir, file_prefix=start_time),
    )

    for epoch in range(TRAIN_ARGS["max_epoch"]):
        assert trainer.epoch == epoch
        trainer.max_epoch = epoch + 1
        result = trainer.run()

        # get extra stats to log to track performance while training
        batch_output = policy(test_collector.buffer)
        logits, actions = batch_output.logits, batch_output.act
        logits_mean = logits.mean(axis=0)
        highest_logit_action = logits.argmax(axis=1)
        values = policy.critic(test_collector.buffer.obs)
        values_mean = values.mean()

        data = dict(
            pmf_logits_mean=logits_mean,
            highest_logit_action=highest_logit_action,
            actions=actions,
            critic_mean_values=values_mean,
            values=values,
        )

        logger.save_extra_data(epoch, data)

    env_info = test_envs.get_env_attr("info", 0)[0]
    _save_result(env_info, result, TRAIN_ARGS, results_dir, start_time)
