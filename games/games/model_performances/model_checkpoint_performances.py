from pyspiel import Game
from games.players.player import RandomPlayer
from games.multiple_match_manager import PlayerF
from pathlib import Path
from functools import partial
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
import argparse

from games.players.value_functions import GAME_TO_VALUE_FUNCTION
from games.players.init_players import init_time_mcts_tf, init_time_alphabeta, init_mcts_random_rollout
from games.model_performances.utils import (
    get_avg_points_results,
    get_results_vs_agents,
    get_time_mcts_checkpoint_player_pool,
    load_game_from_az_dir,
)


def _parse_args() -> tuple[Path, Path, int, int, int, float, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--az_dir", type=str, help="The dir with the AlphaZero Checkpoints")
    parser.add_argument("--save_dir", type=str, help="The dir in which to save the results")
    parser.add_argument("--checkpoint_step_size", type=int, help="AZ Bot evaluated at every n checkpoints")
    parser.add_argument("--nr_games_per_matchup", type=int, help="Avg number of games each checkpoint plays")
    parser.add_argument(
        "--nr_games_vs_opponents", type=int, default=50, help="Nr games played vs players with both with and black"
    )
    parser.add_argument("--time_budget", type=float, default=0.5, help="Nr of seconds per move")
    parser.add_argument("--nr_processes", type=int, default=16, help="Nr processes to run in parallel")

    args = parser.parse_args()
    az_dir = Path(args.az_dir)
    save_dir = Path(args.save_dir)
    assert not save_dir.exists() or not any(save_dir.iterdir())

    checkpoint_step_size = args.checkpoint_step_size
    nr_games_per_matchup = args.nr_games_per_matchup
    nr_games_vs_opponents = args.nr_games_vs_opponents
    time_budget = args.time_budget
    nr_processes = args.nr_processes
    return (
        az_dir,
        save_dir,
        checkpoint_step_size,
        nr_games_per_matchup,
        nr_games_vs_opponents,
        time_budget,
        nr_processes,
    )


def _get_avg_points_results(
    game: Game, az_dir: Path, save_dir: Path, checkpoint_step_size: int, nr_games_per_matchup: int, nr_processes: int
) -> None:
    player_pool = get_time_mcts_checkpoint_player_pool(az_dir=az_dir, checkpoint_step_size=checkpoint_step_size)
    get_avg_points_results(
        game=game,
        save_dir=save_dir,
        player_pool=player_pool,
        nr_games_per_matchup=nr_games_per_matchup,
        nr_processes=nr_processes,
        var="iteration_nr",
    )


def _get_results_vs_agents(
    game: PlayerNrGameWrapper,
    az_dir: Path,
    save_dir: Path,
    nr_games_per_matchup: int,
    time_budget: float,
    nr_processes: int,
) -> None:
    player_f = partial(init_time_mcts_tf, az_path=az_dir)
    agents_fs = _get_benchmark_agent_functions(time_budget=time_budget)
    get_results_vs_agents(
        game=game,
        player_f=player_f,
        agents_fs=agents_fs,
        save_dir=save_dir,
        nr_games_per_matchup=nr_games_per_matchup,
        nr_processes=nr_processes,
    )


def _get_benchmark_agent_functions(time_budget: float) -> list[PlayerF]:
    mcts_player_f = partial(init_mcts_random_rollout, game._game, time_budget)
    agents = [mcts_player_f, RandomPlayer]

    game_name = game.name
    if game_name in GAME_TO_VALUE_FUNCTION:
        alpha_beta_f = partial(init_time_alphabeta, GAME_TO_VALUE_FUNCTION[game_name], time_budget)
        agents.append(alpha_beta_f)

    return agents


if __name__ == "__main__":
    (
        az_dir,
        save_dir,
        checkpoint_step_size,
        nr_games_per_matchup,
        nr_games_per_matchup,
        time_budget,
        nr_processes,
    ) = _parse_args()
    game = load_game_from_az_dir(az_dir)
    _get_avg_points_results(
        game=game,
        az_dir=az_dir,
        save_dir=save_dir,
        checkpoint_step_size=checkpoint_step_size,
        nr_games_per_matchup=nr_games_per_matchup,
        nr_processes=nr_processes,
    )
    _get_results_vs_agents(
        game=game,
        az_dir=az_dir,
        save_dir=save_dir,
        nr_games_per_matchup=nr_games_per_matchup,
        time_budget=time_budget,
        nr_processes=nr_processes,
    )
