from __future__ import annotations
from pyspiel import load_game, Game, State
from math import prod


class PlayerNrGameWrapper:
    def __init__(self, game: Game):
        self._game = game

    def new_initial_state(self):
        state = self._game.new_initial_state()
        return PlayerNrStateWrapper(state, self)

    def observation_tensor_shape(self):
        game_shape = self._game.observation_tensor_shape()
        # Add player nr channel
        game_shape[0] += 1
        return game_shape

    def observation_tensor_size(self):
        return prod(self.observation_tensor_shape())

    def channel_size(self):
        return prod(self.observation_tensor_shape()[1:])

    def __str__(self) -> str:
        return str(self._game)

    def num_distinct_actions(self) -> int:
        return self._game.num_distinct_actions()

    def num_players(self) -> int:
        return self._game.num_players()

    def min_utility(self) -> float:
        return self._game.min_utility()

    def max_utility(self) -> float:
        return self._game.max_utility()

    def get_type(self):
        return self._game.get_type()

    def max_game_length(self):
        return self._game.max_game_length()

    @property
    def name(self) -> str:
        return str(self)[:-2]

    @classmethod
    def from_game_name(cls, game_name: str) -> PlayerNrGameWrapper:
        game = load_game(game_name)
        return cls(game)


class PlayerNrStateWrapper:
    def __init__(self, state: State, game: PlayerNrGameWrapper):
        self.state = state
        self.game = game
        self.channel_size = game.channel_size()

    def get_game(self) -> PlayerNrGameWrapper:
        return self.game

    def clone(self):
        new_state = self.state.clone()
        return PlayerNrStateWrapper(state=new_state, game=self.game)

    def observation_string(self):
        state_string = self.state.observation_string()
        state_string += f"Player to move {self.state.current_player()}\n"
        return state_string

    def observation_tensor(self):
        observation_tensor = self.state.observation_tensor()
        color_channel = [self.state.current_player()] * self.channel_size
        observation_tensor.extend(color_channel)
        return observation_tensor

    def move_number(self) -> int:
        return self.state.move_number()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def current_player(self) -> int:
        return self.state.current_player()

    def legal_actions(self, player_nr: int | None = None) -> list[int]:
        if player_nr is None:
            return self.state.legal_actions()

        return self.state.legal_actions(player_nr)

    def apply_action(self, action: int):
        self.state.apply_action(action)

    def rewards(self) -> list[float]:
        return self.state.rewards()

    def player_return(self, player_nr: int) -> float:
        return self.state.player_return(player_nr)

    def is_chance_node(self) -> bool:
        return self.state.is_chance_node()

    def action_to_string(self, current_player: int, action: int) -> str:
        return self.state.action_to_string(current_player, action)

    def to_string(self) -> str:
        return self.state.to_string()

    def returns(self) -> list[float]:
        return self.state.returns()

    def legal_actions_mask(self):
        return self.state.legal_actions_mask()
