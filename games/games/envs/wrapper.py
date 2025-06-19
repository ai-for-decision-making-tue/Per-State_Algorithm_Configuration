import gymnasium as gym
import numpy as np


class ContinuousToSpecifiedActions(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continuous environment.

    :param gym.Env env: gym environment with continuous action space.
    :param action_per_dim: number of discrete actions in each dimension
        of the action space.
    """

    def __init__(self, env: gym.Env, actions_per_dim: list[list[float]]) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        low_per_dim, high_per_dim = env.action_space.low, env.action_space.high
        if isinstance(low_per_dim, int) and isinstance(high_per_dim, int):
            low_per_dim = [low_per_dim]
            high_per_dim = [high_per_dim]
        elif isinstance(low_per_dim, int) or isinstance(high_per_dim, int):
            raise ValueError("low and high must be either both int or both list.")

        assert isinstance(actions_per_dim[0], list), "actions_per_dim must be a list of lists."

        assert all(
            low <= a <= high for low, high, actions in zip(low_per_dim, high_per_dim, actions_per_dim) for a in actions
        )
        assert len(actions_per_dim) == env.action_space.shape[0]

        self.action_space = gym.spaces.MultiDiscrete([len(actions) for actions in actions_per_dim])

        self.actions_per_dim = actions_per_dim

    def action(self, act: np.ndarray) -> np.ndarray:  # type: ignore
        # modify act
        assert len(act.shape) <= 1, f"Unknown action format with shape: {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.actions_per_dim[i][a] for i, a in enumerate(act)])
        else:
            assert len(self.actions_per_dim) == 1, "act.shape == 0 but actions_per_dim multi-dimensional."
            return np.array([self.actions_per_dim[0][act]])
