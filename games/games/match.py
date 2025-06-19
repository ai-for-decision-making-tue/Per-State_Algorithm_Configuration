from dataclasses import dataclass, field
from pyspiel import Game
from games.players.player import Player
from typing import Optional
import numpy as np
from random import choice


@dataclass
class ObsActionReward:
    state: list[float]
    action: Optional[str] = field(default=None)
    reward: Optional[float] = field(default=np.nan)


class Match:
    """Represents a 2-player match, between player_0, playing color 0 and player_1, playing color 1.

    Args:
        game (Game): The game to be player
        player_0 (Player): The player playing color 0
        player_1 (Player): The player playing color 1
        save_trajectory (bool, optional): Whether to save the trajectory, which can then be obtained with
            .get_trajectory. Defaults to False.
        nr_random_moves (int, optional): The number of random moves to be played before the match starts. Defaults to 0.
    """

    NR_PLAYERS = 2
    PLAYER_NRS = (0, 1)

    def __init__(
        self, game: Game, player_0: Player, player_1: Player, save_trajectory: bool = False, nr_random_moves: int = 0
    ):
        self.game = game
        self.players = [player_0, player_1]
        self.save_trajectory = save_trajectory
        if self.save_trajectory:
            self._trajectory: list[ObsActionReward] = []

        self.nr_random_moves = nr_random_moves
        # self._move_history: list[int] = []

    @classmethod
    def is_first_player_nr(cls, player_nr: int) -> bool:
        return player_nr == cls.PLAYER_NRS[0]

    def reset(self) -> None:
        self.state = self.game.new_initial_state()
        if self.save_trajectory:
            self._trajectory.clear()
        # self._move_history.clear()

        self._play_random_moves()

    def _play_random_moves(self) -> None:
        for _ in range(self.nr_random_moves):
            self._play_random_move()

    def _play_random_move(self) -> None:
        legal_moves = self.state.legal_actions()
        random_move = choice(legal_moves)
        self._apply_move(random_move)

    def play_match(self) -> None:
        while not self.is_terminal:
            self.play_move()

    def play_move(self) -> None:
        # self._rebuild_state()

        move = self.current_player.get_move(self.state)
        if self.save_trajectory:
            obs: list[float] = self.get_observation()
            action = self.state.action_to_string(move)

        self._apply_move(move)

        if self.save_trajectory:
            reward = self.reward
            self._save_state_action_reward(obs, action, reward)

    @property
    def reward(self) -> float:
        # Reward from perspective of player 0, assumes zero-sum games
        return self.state.rewards()[0]

    def _apply_move(self, move: int) -> None:
        self.state.apply_action(move)

    def _save_state_action_reward(self, obs: list[float], action: str, reward: float) -> None:
        oar = ObsActionReward(obs, action, reward)
        self._trajectory.append(oar)

    @property
    def player_0(self) -> Player:
        return self.players[0]

    @property
    def player_1(self) -> Player:
        return self.players[1]

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_nr]

    @property
    def current_player_nr(self) -> int:
        return self.state.move_number() % self.NR_PLAYERS

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def get_trajectory(self) -> list[ObsActionReward]:
        current_obs = self.get_observation()
        current_oar = ObsActionReward(current_obs)
        return self._trajectory + [current_oar]

    def get_observation(self) -> list[float]:
        if self.is_terminal:
            game_name = self.game.name
            # PySpiel does not return the state in numpy version because it's no player's move but we require this data
            # so therefore we created a _get_last_state_from_{game} function to convert the string (which we can get) to
            # the game state.
            if game_name == "connect_four":
                current_state = _get_last_state_from_connect_four_str(self.state)
            elif game_name == "breakthrough":
                current_state = _get_last_state_from_breakthrough_str(self.state)
            else:
                raise ValueError(f"Obtaining last state not implemented for game: {game_name}")
        else:
            current_state = self.state.observation_tensor()

        return current_state

    def set_player(self, nr: int, player: Player) -> None:
        self.players[nr] = player

    def get_player(self, nr: int) -> Player:
        return self.players[nr]

    @property
    def board_string(self) -> str:
        return self.state.to_string()


def _get_last_state_from_connect_four_str(state) -> list[float]:
    symbols = ["o", "x", "."]
    state_elements = [element for element in state.to_string() if element != "\n"]
    numeric_state = [float(element == symbol) for symbol in symbols for element in state_elements if element != "\n"]
    numeric_state.extend(len(state_elements) * [state.current_player()])
    return numeric_state


def _get_last_state_from_breakthrough_str(state) -> list[float]:
    symbols = ["b", "w", "."]
    # last element is ' abcdefgh' which indicates the columns which is not relevant for the state
    relevant_rows = state.to_string().splitlines()[:-1]
    # First element of row is the row number which is not relevant for the state
    state_elements = [element for row in relevant_rows for element in row[1:]]
    numeric_state = [float(element == symbol) for symbol in symbols for element in state_elements if element != "\n"]
    numeric_state.extend(len(state_elements) * [state.current_player()])
    return numeric_state
