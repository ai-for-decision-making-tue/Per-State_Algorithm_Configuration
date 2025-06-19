from __future__ import annotations
from games.players.mcts import TimeMCTS, MCTS, ConfigurationPlayer
from games.players.alpha_beta import TimeAlphaBeta, AlphaBeta
from games.players.player import Player
from pathlib import Path
from typing import Optional, Callable, Any, TYPE_CHECKING
from pyspiel import Game
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

if TYPE_CHECKING:
    from games.utils import MCTSF


def init_configuration_mcts_player(mcts_f: MCTSF, ppo_path: str | Path, c_list: list[float]) -> Player:
    mcts = mcts_f()
    return ConfigurationPlayer.from_path(ppo_path, mcts, c_list=c_list)


def init_time_mcts_tf(
    az_path: str | Path,
    time_budget: float,
    uct_c: Optional[float] = None,
    checkpoint_nr: Optional[int] = None,
    name: Optional[str] = None,
) -> Player:
    return TimeMCTS.load_tensorflow_az(
        az_path, time_budget=time_budget, uct_c=uct_c, checkpoint_nr=checkpoint_nr, name=name
    )


def init_mcts_tf(
    az_path: str | Path,
    max_simulations: int,
    uct_c: Optional[float] = None,
    checkpoint_nr: Optional[int] = None,
    name: Optional[str] = None,
) -> Player:
    return MCTS.load_tensorflow_az(
        az_path, max_simulations=max_simulations, uct_c=uct_c, checkpoint_nr=checkpoint_nr, name=name
    )


def init_alphabeta(value_function: Callable[[Any], float], depth: int, name: str) -> AlphaBeta:
    return AlphaBeta(value_function=value_function, depth=depth, name=name)


def init_time_alphabeta(value_function: Callable[[Any], float], time_budget: float) -> TimeAlphaBeta:
    return TimeAlphaBeta(time_budget=time_budget, value_function=value_function, name="alpha_beta")


def init_mcts_random_rollout(game: Game, time_budget) -> TimeMCTS:
    return TimeMCTS(
        game=game,
        time_budget=time_budget,
        uct_c=2.0,
        evaluator=RandomRolloutEvaluator(),
        name="mcts_random_rollout_evaluator",
    )
