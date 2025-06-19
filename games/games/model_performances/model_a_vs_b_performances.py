from pyspiel import Game
from pathlib import Path

import argparse
from games.model_performances.utils import (
    get_avg_points_results,
    get_time_mcts_checkpoint_player_pool,
    load_game_from_az_dir,
)


def _parse_args() -> tuple[Path, Path, Path, int, int, float, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--az_dir1", type=str, help="The dir of player 1 with the AlphaZero Checkpoints")
    parser.add_argument("--az_dir2", type=str, help="The dir of player 2 with the AlphaZero Checkpoints")
    parser.add_argument("--save_dir", type=str, help="The dir in which to save the results")
    parser.add_argument("--checkpoint_step_size", type=int, help="AZ Bot evaluated at every n checkpoints")
    parser.add_argument("--nr_games_per_matchup", type=int, help="Avg number of games each checkpoint plays")
    parser.add_argument("--time_budget", type=float, default=0.5, help="Nr of seconds per move")
    parser.add_argument("--nr_processes", type=int, default=16, help="Nr processes to run in parallel")

    args = parser.parse_args()
    az_dir1 = Path(args.az_dir1)
    az_dir2 = Path(args.az_dir2)
    save_dir = Path(args.save_dir)
    assert not save_dir.exists() or not any(save_dir.iterdir())

    checkpoint_step_size = args.checkpoint_step_size
    nr_games_per_matchup = args.nr_games_per_matchup
    time_budget = args.time_budget
    nr_processes = args.nr_processes

    return az_dir1, az_dir2, save_dir, checkpoint_step_size, nr_games_per_matchup, time_budget, nr_processes


def _get_avg_points_results(
    game: Game,
    az_dir1: Path,
    az_dir2: Path,
    save_dir: Path,
    checkpoint_step_size: int,
    nr_games_per_matchup: int,
    nr_processes: int,
) -> None:
    player_pool1 = get_time_mcts_checkpoint_player_pool(az_dir=az_dir1, checkpoint_step_size=checkpoint_step_size)
    player_pool2 = get_time_mcts_checkpoint_player_pool(az_dir=az_dir2, checkpoint_step_size=checkpoint_step_size)
    player_pool = player_pool1 + player_pool2

    get_avg_points_results(
        game=game,
        player_pool=player_pool,
        save_dir=save_dir,
        nr_games_per_matchup=nr_games_per_matchup,
        nr_processes=nr_processes,
        var="iteration_nr",
    )


if __name__ == "__main__":
    (
        az_dir1,
        az_dir2,
        save_dir,
        checkpoint_step_size,
        nr_games_per_matchup,
        time_budget,
        nr_processes,
    ) = _parse_args()

    game = load_game_from_az_dir(az_dir1)
    _get_avg_points_results(
        game=game,
        az_dir1=az_dir1,
        az_dir2=az_dir2,
        save_dir=save_dir,
        checkpoint_step_size=checkpoint_step_size,
        nr_games_per_matchup=nr_games_per_matchup,
        nr_processes=nr_processes,
    )
