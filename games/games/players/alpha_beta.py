from dataclasses import dataclass
from typing import Callable, Any, Optional
from time import time
from itertools import count
from games.players.player import Player
from games.players.timeout_exception import TimeOutException


@dataclass
class AlphaBeta(Player):
    depth: int
    value_function: Callable[[Any], float]
    maximizing_player_id = 0
    name: str

    def get_move(self, state) -> int:
        self.start_time = time()
        _, best_move = self._apply_alpha_beta(state=state, depth=self.depth, alpha=-float("inf"), beta=float("inf"))
        return best_move

    def _apply_alpha_beta(self, state, depth: int, alpha: float, beta: float) -> tuple[int, int]:
        # Algorithm adapted from
        # https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/minimax.py
        """An alpha-beta algorithm.

        Implements a min-max algorithm with alpha-beta pruning.
        See for example https://en.wikipedia.org/wiki/Alpha-beta_pruning

        Arguments:
        state: The current state node of the game.
        depth: The maximum depth for the min/max search.
        alpha: best value that the MAX player can guarantee (if the value is <= than
            alpha, the MAX player will avoid it).
        beta: the best value that the MIN currently can guarantee (if the value is
            >= than beta, the MIN player will avoid it).

        Returns:
        A tuple of the optimal value of the sub-game starting in state
        (given alpha/beta) and the move that achieved it.

        Raises:
        NotImplementedError: If we reach the maximum depth. Given we have no value
            function for a non-terminal node, we cannot break early.
        """

        if state.is_terminal():
            return state.player_return(self.maximizing_player_id), None

        if depth == 0:
            return self.value_function(state), None

        player = state.current_player()
        best_action = -1
        if player == self.maximizing_player_id:
            value = -float("inf")
            for action in state.legal_actions():
                child_state = state.clone()
                child_state.apply_action(action)
                child_value, _ = self._apply_alpha_beta(child_state, depth - 1, alpha, beta)
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta cut-off
            return value, best_action
        else:
            value = float("inf")
            for action in state.legal_actions():
                child_state = state.clone()
                child_state.apply_action(action)
                child_value, _ = self._apply_alpha_beta(child_state, depth - 1, alpha, beta)
                if child_value < value:
                    value = child_value
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break  # alpha cut-off
            return value, best_action

    @property
    def nr_iterations(self):
        return self.depth

    def __str__(self) -> str:
        if self.name is not None:
            return f"AlphaBeta[depth={self.depth}, name={self.name}]"

        return f"AlphaBeta(depth={self.depth}, value_function={self.value_function})"


class TimeAlphaBeta(AlphaBeta):
    def __init__(self, time_budget: float, value_function: Callable[[Any], float], name: Optional[str] = None):
        self.time_budget = time_budget
        self.value_function = value_function
        self.name = name

    def get_move(self, state) -> int:
        self.start_time = time()
        try:
            for depth in count(start=1):
                self.depth = depth
                _, move = self._apply_alpha_beta(state=state, depth=depth, alpha=-float("inf"), beta=float("inf"))
        except TimeOutException:
            self.depth -= 1

        return move

    def _apply_alpha_beta(self, state, depth: int, alpha: float, beta: float) -> tuple[int, Optional[int]]:
        if self._is_out_of_time():
            raise TimeOutException

        if state.is_terminal():
            return state.player_return(self.maximizing_player_id), None

        return super()._apply_alpha_beta(state, depth, alpha, beta)

    def _is_out_of_time(self):
        current_time = time()
        time_passed = current_time - self.start_time
        return time_passed > self.time_budget

    def __str__(self) -> str:
        if self.name is not None:
            return f"TimeAlphaBeta[time_budget={self.time_budget}, name={self.name}]"

        return f"TimeAlphaBeta(time_budget={self.time_budget}, value_function={self.value_function})"
