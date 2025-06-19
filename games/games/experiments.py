from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional, Callable
from games.players.init_players import init_mcts_tf, init_configuration_mcts_player
from functools import partial
from games.envs.game_env import BaseGameEnvironment, ConfigurationGameEnvironment
from pathlib import Path
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
from games.envs.wrapper import ContinuousToSpecifiedActions


if TYPE_CHECKING:
    from games.utils import PlayerF, MCTSF


@dataclass(kw_only=True)
class ExperimentParams(ABC):
    name: str
    game_name: str
    # baselines: dict
    ppo_file_name: Optional[str] = None  # To be filled in after Reinfocement Learning Experiment
    nr_random_moves: int = 2

    @property
    @abstractmethod
    def experiment_type(self) -> str:
        pass

    @abstractmethod
    def get_player_pool(self) -> list[PlayerF]:
        pass

    @abstractmethod
    def create_env_f(self) -> Callable[[], BaseGameEnvironment]:
        pass

    @property
    def ppo_file_path(self) -> Optional[Path]:
        if self.ppo_file_name is None:
            return None

        return self.results_dir_path / self.ppo_file_name

    @property
    def results_dir_path(self) -> Path:
        return Path("results") / self.experiment_type / self.game_name / self.name

    @property
    def tensorboard_dir_path(self) -> Path:
        return Path("tensorboard_logs") / self.experiment_type / self.game_name / self.name


@dataclass(kw_only=True)
class ConfigurationExperimentParams(ExperimentParams):
    c_values: list[float]
    player_f: MCTSF
    best_c: Optional[float] = None  # To be filled in after Tweak hyperparameters

    def create_env_f(self, core_nr: int) -> BaseGameEnvironment:
        game = PlayerNrGameWrapper.from_game_name(self.game_name)
        env = ConfigurationGameEnvironment(
            game=game,
            player_f=self.player_f,
            core_nr=core_nr,
            nr_random_moves=self.nr_random_moves,
            c_values=self.c_values,
        )
        env = ContinuousToSpecifiedActions(env, [self.c_values])
        return env

    def get_best_static_player(self) -> PlayerF:
        if self.best_c is None:
            raise ValueError("Best c value is None")

        return self.get_player(c=self.best_c, name="best_static")

    def get_player(self, c: float, name: Optional[str] = None) -> PlayerF:
        return partial(self.player_f, uct_c=c, name=name)

    def get_ppo_player(self) -> PlayerF:
        assert self.ppo_file_name is not None, "PPO file name is None"
        return partial(
            init_configuration_mcts_player, mcts_f=self.player_f, ppo_path=self.ppo_file_path, c_list=self.c_values
        )

    def get_player_pool(self) -> list[PlayerF]:
        return [self.get_player(c) for c in self.c_values]

    @property
    def experiment_type(self) -> str:
        return "AlgorithmConfiguration"


CONNECT_FOUR_TIMEOUT = 0.5
CONNECT_FOUR_NAME = "connect_four"
PLAYER_F_CONNECT_FOUR = partial(init_mcts_tf, az_path="models/connect_four_long_run", max_simulations=100)

CONNECT_FOUR_CONFIGURATION_EXP_C = ConfigurationExperimentParams(
    name="connect_four_configuration_exp_c",
    game_name=CONNECT_FOUR_NAME,
    c_values=[2**i for i in range(-7, 3)],
    player_f=PLAYER_F_CONNECT_FOUR,
    best_c=0.25,
    ppo_file_name="2024_12_02_15_56_ppo.pt",
)


EXPERIMENTS = {
    exp.name: exp
    for exp in [
        CONNECT_FOUR_CONFIGURATION_EXP_C,
    ]
}
