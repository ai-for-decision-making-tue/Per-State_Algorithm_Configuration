from __future__ import annotations
import argparse
import json
from pyspiel import Game
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from games.experiments import ExperimentParams
    from games.players.player import Player
    from games.players.mcts import MCTS

    PlayerF = Callable[[], Player]
    MCTSF = Callable[[], MCTS]


def parse_experiment_params() -> tuple[ExperimentParams, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="The experiment that will be run, e.g. 'breakthrough_8'")
    args, unparsed_args = parser.parse_known_args()
    experiment_params = get_experiment_params(args.experiment)
    return experiment_params, unparsed_args


def get_experiment_params(experiment_name: str) -> ExperimentParams:
    # moved inside function to avoid circular imports
    from games.experiments import EXPERIMENTS

    return EXPERIMENTS[experiment_name]


def load_json(path) -> dict:
    with open(path) as f:
        data_dict = json.load(f)

    return data_dict


def get_game_name(game: Game) -> str:
    return str(game)[:-2]
