import pandas as pd
from pathlib import Path
from itertools import pairwise

import argparse

BASELINES = dict(AlwaysTrue=[True], AlwaysFalse=[False], AlternatingTrueFalse=[True, False])
BASE_BASELINES_DIR = Path("results/baselines")
EXPECTED_LENGTHS = {"Breakthrough-8": 90, "Knightthrough-6": 118}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Breakthrough-8", help="The experiment that will be run")
    parser.add_argument(
        "--datetime", type=str, default="2024_08_13_16_34", help="Datetime the experiment was carried out"
    )
    parser.add_argument("--baseline", type=str, default="AlwaysTrue", help="Name of the baseline whose games to verify")
    args = parser.parse_args()
    return args.game, args.datetime, args.baseline


def validate_games(games_dir: Path, actions: list[bool], game_name: str):
    for game_nr, game_path in enumerate(games_dir.iterdir(), start=1):
        game = pd.read_csv(game_path)
        validate_single_game(game, actions=actions, game_name=game_name)

    return game_nr


def validate_single_game(game: pd.DataFrame, actions: list[bool], game_name: str):
    validate_positions(game["position"], game_name)
    validation_actions(game["action"], actions)
    validate_players(game["player"], actions)
    validate_opponents(game["opponent"])
    validate_player_colors(game["player_color"])
    validate_colors_to_play(game["color_to_play"], game["reward"].iloc[-1])
    validate_dones(game["done"])
    validate_rewards(game["reward"])


def validate_positions(positions: pd.Series, game_name: str):
    expected_length = EXPECTED_LENGTHS[game_name]

    assert (positions.apply(len) == expected_length).all(), "All positions should be 182 chars long"
    positions = positions.str[:-20]  ## Strip color info
    for pos1, pos2 in pairwise(positions):
        nr_differences = _get_nr_differences_two_strings(pos1, pos2)
        assert nr_differences <= 4, "Too many changes after both players moved"


def validation_actions(action: pd.Series, actions: list[bool]):
    assert action.isna().iloc[-1], "Final action needs to be NA"

    if len(actions) == 1:
        assert (action.iloc[:-1] == actions[0]).all(), "Unexpected action"
        return

    if action.iloc[0] != actions[0]:
        actions = [actions[1], actions[0]]

    len_remaining_actions = len(action) - 1
    nr_full_actions_repeats = len_remaining_actions // 2
    assert (action.iloc[: nr_full_actions_repeats * 2] == actions * nr_full_actions_repeats).all(), "Unexpected action"
    if len_remaining_actions % 2 == 1:
        assert action.iloc[-2] == actions[0], "Unexpected action"


def validate_players(player: pd.Series, actions: list[bool]):
    assert player.isna().iloc[-1], "Final player needs to be NA"

    if len(actions) == 1:
        assert len(player.iloc[:-1].unique()) == 1, "Expected 1 unique player"
        return

    if player.iloc[0] != actions[0]:
        actions = [player[1], player[0]]

    len_remaining_actions = len(player) - 1
    nr_full_actions_repeats = len_remaining_actions // 2

    expected_repeats = player[:2].to_list() * nr_full_actions_repeats
    assert (player.iloc[: nr_full_actions_repeats * 2] == expected_repeats).all(), "Unexpected player"
    if len_remaining_actions % 2 == 1:
        assert player.iloc[-2] == player.iloc[0], "Unexpected action"


def validate_opponents(opponent: pd.Series):
    assert len(opponent.unique()) == 1, "Expected one different opponent"


def validate_player_colors(player_color: pd.Series):
    assert len(player_color.unique()) == 1, "Expected player color to be the same"


def _get_nr_differences_two_strings(s1: str, s2: str) -> int:
    return sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))


def validate_colors_to_play(color_to_play: pd.Series, last_reward: float):
    unique_colors_excl_last_move = color_to_play.iloc[:-1].unique()
    assert len(unique_colors_excl_last_move) == 1


def validate_dones(done: pd.Series):
    assert done.isna().iloc[0], "Expected first state to be done"
    assert (done.iloc[1:-1] == False).all(), "Expected states before last to not be done"
    assert done.iloc[-1] == True, "Expected last state to be done"


def validate_rewards(reward: pd.Series):
    is_na = reward.isna()
    assert is_na.iloc[0], "first entry not NA"
    assert ~is_na.iloc[1:].any(), "At least one entry after first NA"
    assert (reward.iloc[1:-1] == 0).all(), "not all rewards[1:-1] 0"


if __name__ == "__main__":
    game, experiment_datetime, baseline = _parse_args()
    actions = BASELINES[baseline]
    games_dir = BASE_BASELINES_DIR / game / experiment_datetime / "games" / baseline
    nr_games = validate_games(games_dir=games_dir, actions=actions, game_name=game)
    print(f"Succesfully validated {nr_games} games, exiting")
