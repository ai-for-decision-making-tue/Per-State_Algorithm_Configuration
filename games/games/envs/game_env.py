from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyspiel import Game
from abc import abstractmethod, ABCMeta
from random import choice
from typing import Optional, Sequence, TYPE_CHECKING
import os
import torch

from games.players.player import Player
from games.players.mcts import MCTS
from games.players.init_players import init_mcts_random_rollout
from games.match import Match
from games.utils import get_game_name
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper

if TYPE_CHECKING:
    from games.utils import MCTSF


class BaseGameEnvironment(gym.Env, metaclass=ABCMeta):
    """The base game environment that defines the meta-MDP.

    Args:
        game (Game): The game to be played.
        player (Player): The player to be used in the environment.
        core_nr (int): The core number to run the environment on, environments are run on single cores to support
            efficient parallelization.
        nr_random_moves (int): The number of random moves to be played at the start of the game.
    """

    def __init__(self, game: Game, player: Player, core_nr: int, nr_random_moves: int):
        self.game = game
        self.player = player
        self.nr_random_moves = nr_random_moves

        self.feature_shape = tuple(game.observation_tensor_shape())
        self.reward_range = (game.min_utility(), game.max_utility())

        self.observation_space = spaces.Dict(
            {"obs": spaces.MultiBinary(self.feature_shape), "mask": spaces.MultiBinary((1))}
        )

        self.nr_iterations: tuple[list[int], list[int]] = ([], [])
        self.core_nr = core_nr

    def reset(self, seed=None, options=None, player_color_nr: Optional[int] = None):
        opponent = self._get_opponent()
        self._reset_match(self.player, opponent, player_color_nr)
        obs = self._get_obs()
        return obs, {}  # second return value is empty info

    @abstractmethod
    def _get_opponent(self) -> Player:
        pass

    def _reset_match(self, player: Player, opponent: Player, player_nr: Optional[int]) -> None:
        os.sched_setaffinity(os.getpid(), {self.core_nr})
        torch.set_num_threads(1)

        if player_nr is None:
            player_nr = choice(Match.PLAYER_NRS)

        self.player_nr = player_nr
        self.opponent_nr = 1 - self.player_nr

        player_0 = player if Match.is_first_player_nr(self.player_nr) else opponent
        player_1 = opponent if Match.is_first_player_nr(self.player_nr) else player

        self.match = Match(game=self.game, player_0=player_0, player_1=player_1, nr_random_moves=self.nr_random_moves)
        self.match.reset()

        for depth in self.nr_iterations:
            depth.clear()

        if not self.is_player_turn():
            self._play_move()

    def is_player_turn(self) -> bool:
        return self.player_nr == self.match.current_player_nr

    def _get_obs(self, is_terminal: bool = False) -> dict:
        obs = self.match.get_observation()
        features = np.array(obs)
        features = features.reshape(self.feature_shape)
        mask = np.array([1])

        obs = {"obs": features, "mask": mask}

        return obs

    def _play_move_given_action(self, action: float | bool) -> None:
        """Returns whether the environment is terminated"""
        self.player = self._get_player_given_action(action=action)
        self.match.set_player(nr=self.player_nr, player=self.player)
        self._play_move()

    @abstractmethod
    def _get_player_given_action(self, action: float | bool) -> Player:
        pass

    def _play_move(self) -> None:
        current_player_nr = self.match.current_player_nr
        current_player = self.current_player
        self.match.play_move()
        self.nr_iterations[current_player_nr].append(current_player.nr_iterations)

    @property
    def is_terminal(self):
        return self.match.is_terminal

    def get_player_string(self, action: bool) -> str:
        player = self._get_player_given_action(action=action)
        return str(player)

    @property
    def opp_string(self):
        opponent = self.match.get_player(self.opponent_nr)
        return str(opponent)

    @property
    def current_player(self):
        return self.match.current_player

    @property
    def board_string(self):
        return self.match.board_string

    def _get_reward(self) -> float:
        # Match stores reward based on player 0 perspective. So if player is player 2 we need to negate.
        return self.match.reward if Match.is_first_player_nr(self.player_nr) else -1 * self.match.reward

    def step(self, action: float | bool):
        os.sched_setaffinity(os.getpid(), {self.core_nr})
        torch.set_num_threads(1)

        # self.match._rebuild_state()
        self._play_move_given_action(action)
        if not self.is_terminal:
            self._play_move()

        obs = self._get_obs(self.is_terminal)
        reward = self._get_reward()
        truncated = False

        return (obs, reward, self.is_terminal, truncated, {})

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    @property
    def player_strings(self) -> list[str]:
        return [str(player) for player in [self.match.player_0, self.match.player_1]]

    @property
    def game_name(self) -> str:
        return get_game_name(self.game)

    @property
    def info(self) -> dict:
        return dict(game=self.game_name, players=self.player_strings)

    @property
    def player_nr_iterations(self) -> list[int]:
        return self.nr_iterations[self.player_nr]

    @property
    def opponent_nr_iterations(self) -> list[int]:
        return self.nr_iterations[self.opponent_nr]

    @property
    def state(self):
        return self.match.state


class PlayerSelectionGameEnvironment(BaseGameEnvironment):
    """A Player Selection Game Environment that defines a meta-MDP, where the action is which player to select.

    Args:
        game (Game): The game to be played.
        players (Player): List of players that can be selected
        core_nr (int): The core number to run the environment on, environments are run on single cores to support
            efficient parallelization.
        nr_random_moves (int): The number of random moves to be played at the start of the game.
    """

    def __init__(self, game: Game, players: Sequence[Player], core_nr: int, nr_random_moves: int):
        # As player will be selected based on action, we can set player to the first player in the list as it will be
        # overwritten before the first move.
        super().__init__(game=game, player=players[0], core_nr=core_nr, nr_random_moves=nr_random_moves)
        self.players = tuple(players)
        self.action_space = spaces.Discrete(len(self.players))

    def _get_opponent(self) -> Player:
        return choice(self.players)

    def _get_player_given_action(self, action: int) -> Player:
        return self.players[action]


class ConfigurationGameEnvironment(BaseGameEnvironment):
    """A Configuration Game Environment that defines a meta-MDP, where the action is how to define the parameters of the
    player. Currently only a MCTS player that defines the UCT constant is supported.

    Args:
        game (Game): The game to be played.
        player_f (MCTSF): The MCTS player_f to be used in the environment.
        core_nr (int): The core number to run the environment on, environments are run on single cores to support
            efficient parallelization.
        nr_random_moves (int): The number of random moves to be played at the start of the game.
    """

    def __init__(self, game: Game, player_f: MCTSF, core_nr: int, nr_random_moves: int, c_values: list[float]):
        player = player_f()
        super().__init__(game, player, core_nr, nr_random_moves)

        self.opponent = player_f()
        self.action_space = spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32)
        self.c_values = c_values

    def _get_opponent(self) -> MCTS:
        self.opponent.uct_c = choice(self.c_values)
        return self.opponent

    def _get_player_given_action(self, action: float | bool) -> MCTS:
        self.player: MCTS  # type: ignore
        self.player.uct_c = action
        return self.player


def print_obs(obs):
    white_obs = obs["obs"][0]
    black_obs = obs["obs"][1] * 2
    print_obs = white_obs + black_obs
    print(print_obs, end="\n\n")


if __name__ == "__main__":
    game = PlayerNrGameWrapper.from_game_name("connect_four")
    mcts_player_10 = init_mcts_random_rollout(game, 4)
    mcts_player_1 = init_mcts_random_rollout(game, 3)
    env = PlayerSelectionGameEnvironment(
        game=game, player_a=mcts_player_10, player_b=mcts_player_1, core_nr=0, nr_random_moves=2
    )
    obs, _ = env.reset(player_color_nr=0)

    print_obs(obs)

    while True:
        _, reward, terminated, truncated, _ = env.step(True)
        if terminated or truncated:
            break

        obs = env._get_obs()
        print_obs(obs)

    print(reward)
