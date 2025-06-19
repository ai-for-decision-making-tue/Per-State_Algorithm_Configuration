from __future__ import annotations
from argparse import ArgumentParser
from torch.multiprocessing import set_start_method
from math import ceil
from typing import TYPE_CHECKING

from games.model_performances.utils import get_avg_points_and_plot, get_nr_games_per_opp_per_colour
from games.utils import parse_experiment_params
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
from games.experiments import ConfigurationExperimentParams

if TYPE_CHECKING:
    from games.utils import PlayerF


def _parse_args() -> tuple[ConfigurationExperimentParams, int, int]:
    experiment, unparsed_args = parse_experiment_params()
    parser = ArgumentParser()
    parser.add_argument("--nr_games", type=int, help="Number of games that each player plays in total")
    parser.add_argument("--nr_processes", type=int, default=16, help="Nr processes to run in parallel")
    args = parser.parse_args(unparsed_args)
    nr_games = args.nr_games
    nr_processes = args.nr_processes

    assert isinstance(experiment, ConfigurationExperimentParams), "This script is only for Configuration Experiments"
    assert experiment.ppo_file_name is not None

    return experiment, nr_games, nr_processes


def _get_match_ups_both_colours(init_player: PlayerF, opponents: list[PlayerF]) -> list[tuple[PlayerF, PlayerF]]:
    return [(init_player, opponent) for opponent in opponents] + [(opponent, init_player) for opponent in opponents]


def _get_avg_points_results(experiment: ConfigurationExperimentParams, nr_games: int, nr_processes: int) -> None:
    game = PlayerNrGameWrapper.from_game_name(experiment.game_name)
    nr_games_per_opp = get_nr_games_per_opp_per_colour(nr_games, nr_opponents=len(experiment.get_player_pool()))
    player_pool = experiment.get_player_pool()
    var = "c"

    players = ["baseline", "ppo"]

    for player in players:
        if player == "baseline":
            player_f = experiment.get_best_static_player()
        else:
            player_f = experiment.get_ppo_player()

        match_ups = _get_match_ups_both_colours(player_f, player_pool)
        save_dir = experiment.results_dir_path / f"model_performances_{player}_vs_random_c"
        assert not save_dir.exists() or not any(save_dir.iterdir())

        get_avg_points_and_plot(game, save_dir, match_ups, nr_random_moves=0, nr_processes=nr_processes, var=var)
        match_ups_2_moves = match_ups * nr_games_per_opp
        get_avg_points_and_plot(
            game, save_dir, match_ups_2_moves, nr_random_moves=2, nr_processes=nr_processes, var=var
        )

    nr_games_players_per_colour = ceil(nr_games / 2)
    player_a = experiment.get_best_static_player()
    player_b = [experiment.get_ppo_player()] * nr_games_players_per_colour
    match_ups = _get_match_ups_both_colours(player_a, player_b)

    save_dir = experiment.results_dir_path / f"model_performances_{players[0]}_vs_{players[1]}"

    get_avg_points_and_plot(game, save_dir, match_ups, nr_random_moves=0, nr_processes=nr_processes, var=var)
    get_avg_points_and_plot(game, save_dir, match_ups, nr_random_moves=2, nr_processes=nr_processes, var=var)


if __name__ == "__main__":
    set_start_method("spawn")
    experiment, nr_games, nr_processes = _parse_args()
    _get_avg_points_results(experiment=experiment, nr_games=nr_games, nr_processes=nr_processes)
