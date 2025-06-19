from abc import abstractmethod, ABCMeta
from random import choice


class Player(metaclass=ABCMeta):
    nr_iterations: int

    @abstractmethod
    def get_move(self, state) -> int: ...


class RandomPlayer(Player):
    nr_iterations = 0

    def get_move(self, state) -> int:
        legal_actions = state.legal_actions(state.current_player())
        random_action = choice(legal_actions)
        return random_action

    def __str__(self) -> str:
        return "RandomPlayer"
