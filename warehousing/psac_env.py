import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding

from dp import dynaplex
from planning.mcts_parametrized_search import MCTS


class PSACEnv(gym.Env):
    def __init__(self, mdp, timestep_delta=10, startup_duration=4000, max_episode_len=10000):

        self.mdp = mdp
        self.timestep_delta = timestep_delta
        self. startup_duration = startup_duration
        self.max_episode_len = max_episode_len

        # Action space is multi_discrete, with as many dimensions as the number of parameters to configure
        self.action_space = spaces.Discrete(20)

        self.n_node_features = 7
        self.n_nodes = 351

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1, high=1000, shape=(self.n_node_features * self.n_nodes,), dtype=np.float64),
                "mask": spaces.Sequence(spaces.Discrete(2)),
            }
        )

        self.policy = self.mdp.get_policy("random")

        self.traj = None
        self.search_traj = None

        self.tree = None

        # Fixed parameters
        self.gamma = 0.99
        self.n_mcts = 250
        self.max_period_count = 200
        self.max_search_time = 10000000000  # 10s

    def reset(self, seed=None, options=None):

        self.traj = dynaplex.get_trajectory(0)
        self.search_traj = dynaplex.get_trajectory(1)

        if seed is None:
            generator, _ = seeding.np_random()
            seed_gen = generator.integers(0, 10000, dtype=np.int64)
            seed = seed_gen.item()

        self.traj.seed_rngprovider(seed)
        self.search_traj.seed_rngprovider(seed)

        self.mdp.initiate_state(self.traj)

        self.mdp.incorporate_until_nontrivial_action(self.traj)
        while self.traj.period_count < self.startup_duration / self.timestep_delta:
            self.mdp.incorporate_action(self.traj, self.policy)    # Apply greedy heuristic policy
            self.mdp.incorporate_until_nontrivial_action(self.traj)

        obs = {'obs': self.mdp.get_features(self.traj)[0],
               'mask': self.mdp.get_mask(self.traj)[0]}

        self.tree = MCTS(root_obs=obs, root=None, gamma=self.gamma, search_env=self.mdp, search_traj=self.search_traj)

        obs['obs'] = torch.from_numpy(obs['obs'])
        obs['mask'] = torch.from_numpy(obs['mask'])

        return obs, {}  # second return value is empty info

    def step(self, action):

        if hasattr(action[0], "__len__"):
            action = (action[0].item())

        c = 0.25
        n_chance_nodes = 10
        policy = self.policy       # random

        # Translate action to MCTS parameter
        max_len = 5 + 5 * action

        start_return = self.traj.cumulative_return

        self.tree.search(n_mcts=self.n_mcts, n_chance_nodes=n_chance_nodes, c=c, policy=policy,
                         start_state=self.traj, start_return=start_return,
                         max_len=max_len, max_time=self.max_search_time)

        pi, V = self.tree.return_results(temp=1.0)  # extract the root output
        child_action_idx = np.argmax(pi)
        mdp_action = self.tree.root.child_actions[child_action_idx].index

        self.traj.next_action = mdp_action
        self.mdp.incorporate_action(self.traj)

        self.mdp.incorporate_until_action(self.traj)

        terminated = self.traj.period_count >= self.max_episode_len
        truncated = False
        reward = - (self.traj.cumulative_return - start_return)

        obs = {'obs': self.mdp.get_features(self.traj)[0],
               'mask': self.mdp.get_mask(self.traj)[0]}
        self.tree.forward(child_action_idx, obs)

        obs['obs'] = torch.from_numpy(obs['obs'])
        obs['mask'] = torch.from_numpy(obs['mask'])

        return (
            obs,
            reward,
            terminated,
            truncated,
            {}
        )

    def render(self, mode='human'):
        pass

    def close(self):
        pass
