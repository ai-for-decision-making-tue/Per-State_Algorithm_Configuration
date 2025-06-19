from __future__ import annotations
import pandas as pd
from pyspiel import load_game, Game
from pathlib import Path
from matplotlib import pyplot as plt
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
from games.multiple_match_manager import MultipleMatchManager
from math import ceil
from itertools import product, count
from functools import partial
from games.utils import load_json
from games.players.init_players import init_time_mcts_tf
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.utils import PlayerF


def load_game_from_az_dir(path: Path | str) -> PlayerNrGameWrapper:
    path = Path(path)
    config_dict = load_json(path / "config.json")
    game = load_game(config_dict["game"])
    game = PlayerNrGameWrapper(game)
    return game


def get_avg_points_results(
    game: Game, player_pool: list[PlayerF], save_dir: Path, nr_games_per_matchup: int, nr_processes: int, var: str
) -> None:
    match_ups = [(p1_f, p2_f) for p1_f, p2_f in product(player_pool, repeat=2)]
    get_avg_points_and_plot(game, save_dir, match_ups, nr_random_moves=0, nr_processes=nr_processes, var=var)

    match_ups_2_moves = match_ups * nr_games_per_matchup
    get_avg_points_and_plot(game, save_dir, match_ups_2_moves, nr_random_moves=2, nr_processes=nr_processes, var=var)


def extract_var(string: str, var: str) -> float:
    match = re.search(rf"{var}=([0-9]*\.?[0-9]+)", string)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not extract {var} from string: {string}")


def get_avg_points_and_plot(
    game: Game,
    save_dir: Path,
    match_ups: list[tuple[PlayerF, PlayerF]],
    nr_random_moves: int,
    nr_processes: int,
    var: str,
) -> None:
    save_dir = save_dir / f"nr_random_moves_{nr_random_moves}"

    mmm = MultipleMatchManager(
        game=game,
        match_ups=match_ups,
        nr_processes=nr_processes,
        nr_random_moves=nr_random_moves,
        save_dir=save_dir,
    )
    results = mmm.play()

    avg_points = get_avg_points(results)
    if var in ["iteration_nr", "c"]:
        avg_points[var] = avg_points["player"].apply(extract_var, var=var)
    else:
        raise ValueError(f"Unexpected value for var: {var}")

    avg_points = avg_points.sort_values(var)

    avg_points.to_csv(save_dir / "avg_points.csv")
    save_plot_avg_points(avg_points, save_dir=save_dir, var=var)


def get_avg_points(results: pd.DataFrame) -> pd.DataFrame:
    long_df = pd.melt(
        results, id_vars=["result"], value_vars=["player0", "player1"], var_name="player_type", value_name="player"
    )
    long_df["points"] = long_df.apply(
        lambda row: row["result"] if row["player_type"] == "player0" else 1 - row["result"], axis=1
    )
    avg_points = long_df.groupby("player", as_index=False)["points"].mean()
    return avg_points


def save_plot_avg_points(avg_points: pd.DataFrame, save_dir: Path, var: str) -> None:
    plt.figure()
    plt.plot(avg_points[var], avg_points["avg_points"], ".")
    plt.savefig(save_dir / f"plot_{var}_vs_avg_points.png", format="png", dpi=300)
    plt.close()


def get_results_vs_agents(
    game: PlayerNrGameWrapper,
    player_f: PlayerF,
    agents_fs: list[PlayerF],
    save_dir: Path,
    nr_games_per_matchup: int,
    nr_processes: int,
) -> None:
    player_results = []

    for opponent_f in agents_fs:
        opponent_str = str(opponent_f())
        vs_opponent_save_dir = save_dir / f"vs_{opponent_str}"
        match_ups = [(player_f, opponent_f), (opponent_f, player_f)] * nr_games_per_matchup
        mmm = MultipleMatchManager(
            game=game, match_ups=match_ups, nr_processes=nr_processes, nr_random_moves=2, save_dir=vs_opponent_save_dir
        )
        results = mmm.play()

        player_avg_player0_result = results.loc[results.player1 == opponent_str, "result"].mean()
        player_avg_player1_result = -1 * results.loc[results.player0 == opponent_str, "result"].mean()
        player_results.append(
            dict(
                opponent=opponent_str,
                avg_result_player0=player_avg_player0_result,
                avg_result_player1=player_avg_player1_result,
            )
        )

    player_results = pd.DataFrame(player_results)
    player_results.to_csv(save_dir / "vs_opponents_results.csv", index=False)


def get_time_mcts_checkpoint_player_pool(az_dir: Path, checkpoint_step_size: int) -> list[PlayerF]:
    max_nr_checkpoints = _get_max_nr_checkpoints(az_dir, checkpoint_step_size)

    player_pool = [
        partial(init_time_mcts_tf, az_dir, checkpoint_nr * checkpoint_step_size)
        for checkpoint_nr in range(max_nr_checkpoints)
    ]
    return player_pool  # type: ignore


def _get_max_nr_checkpoints(path: Path | str, checkpoint_step_size: int) -> int:
    path = Path(path)

    for nr_checkpoints in count(start=1):
        checkpoint_nr = nr_checkpoints * checkpoint_step_size
        checkpoint_path = path / f"checkpoint-{checkpoint_nr}.meta"

        if not checkpoint_path.exists():
            return nr_checkpoints


def get_nr_games_per_opp_per_colour(nr_games: int, nr_opponents: int) -> int:
    return ceil(nr_games / nr_opponents / 2)
