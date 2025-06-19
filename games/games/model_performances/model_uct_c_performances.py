from argparse import ArgumentParser

from games.model_performances.utils import get_avg_points_results, get_nr_games_per_opp_per_colour
from games.utils import parse_experiment_params
from games.experiments import ConfigurationExperimentParams
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper


def _parse_args() -> tuple[ConfigurationExperimentParams, int, int]:
    experiment, unparsed_args = parse_experiment_params()
    parser = ArgumentParser()
    parser.add_argument("--nr_games", type=int, help="Number of games that each player plays in total")
    parser.add_argument("--nr_processes", type=int, default=16, help="Nr processes to run in parallel")
    args = parser.parse_args(unparsed_args)
    nr_games = args.nr_games
    nr_processes = args.nr_processes

    return experiment, nr_games, nr_processes


def _get_avg_points_results(experiment: ConfigurationExperimentParams, nr_games: int, nr_processes: int) -> None:
    game = PlayerNrGameWrapper.from_game_name(experiment.game_name)
    player_pool = experiment.get_player_pool()
    nr_games_per_opp_per_colour = get_nr_games_per_opp_per_colour(nr_games=nr_games, nr_opponents=len(player_pool))
    save_dir = experiment.results_dir_path / "model_performances_c_values"
    assert not save_dir.exists() or not any(save_dir.iterdir())

    get_avg_points_results(
        game=game,
        save_dir=save_dir,
        player_pool=player_pool,
        nr_games_per_matchup=nr_games_per_opp_per_colour,
        nr_processes=nr_processes,
        var="c",
    )


if __name__ == "__main__":
    experiment, nr_games, nr_processes = _parse_args()
    _get_avg_points_results(experiment=experiment, nr_games=nr_games, nr_processes=nr_processes)
