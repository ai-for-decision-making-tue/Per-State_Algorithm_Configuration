from __future__ import annotations
from games.match import Match
import torch
import os
from torch.multiprocessing import Pool
from dataclasses import dataclass
from pyspiel import Game
import pandas as pd
from pathlib import Path
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
from math import ceil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from games.utils import PlayerF


@dataclass
class MultipleMatchManager:
    game: PlayerNrGameWrapper
    match_ups: list[tuple[PlayerF, PlayerF]]
    nr_processes: int
    nr_random_moves: int
    save_dir: Path

    def play(self) -> pd.DataFrame:
        self.save_dir.mkdir(exist_ok=True, parents=True)

        core_nrs_available = os.sched_getaffinity(0)
        assert len(core_nrs_available) >= self.nr_processes, "too few cores, cannot assign unique core to each process"
        SIZE = self.nr_processes * 5

        results = []

        for i in range(ceil(len(self.match_ups) / SIZE)):
            pool = Pool(processes=self.nr_processes)
            pid_to_core = {process.pid: core_nr for core_nr, process in zip(core_nrs_available, pool._pool)}

            start_chunk = i * SIZE
            args_chunk = [
                (self.game._game, matchup, self.nr_random_moves, self.save_dir, start_chunk + match_nr, pid_to_core)
                for match_nr, matchup in enumerate(self.match_ups[start_chunk : start_chunk + SIZE])
            ]
            results_chunk = pool.starmap(_play, args_chunk)

            pool.close()

            results_chunk = pd.DataFrame(results_chunk, columns=["match_nr", "player0", "player1", "result"])
            results.append(results_chunk)

        results = pd.concat(results)
        results.to_csv(self.save_dir / "results.csv", index=False)
        return results


def _play(
    game: Game,
    matchup: tuple[PlayerF, PlayerF],
    nr_random_moves: int,
    save_dir: Path,
    match_nr: int,
    pid_to_core: dict[int, int],
) -> tuple[int, str, str, float]:
    pid = os.getpid()
    os.sched_setaffinity(pid, {pid_to_core[pid]})
    torch.set_num_threads(1)
    game = PlayerNrGameWrapper(game=game)

    player_0 = matchup[0]()
    player_1 = matchup[1]()
    match = Match(game, player_0=player_0, player_1=player_1, save_trajectory=True, nr_random_moves=nr_random_moves)

    match.reset()
    match.play_match()

    trajectory = match.get_trajectory()
    df = pd.DataFrame(trajectory)
    save_path = save_dir / f"match_{match_nr}.parquet"
    df.to_parquet(save_path)

    result = match.reward
    player_0_str, player_1_str = str(player_0), str(player_1)
    return (match_nr, player_0_str, player_1_str, result)
