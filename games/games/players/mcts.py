from __future__ import annotations
from dataclasses import dataclass
from games.players.player import Player
import pyspiel
from pyspiel import Game, load_game
from games.train_open_spiel_player_nr.player_nr_game_wrapper import PlayerNrGameWrapper
from open_spiel.python.algorithms.mcts import SearchNode, Evaluator
from games.networks.alpha_zero.torch import AZModel, load_az_model
from games.networks.alpha_zero.tensorflow import Model as TFModel
from pathlib import Path
from games.utils import load_json
import numpy as np
from typing import Optional, Callable
from time import time
from itertools import count
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AZEvaluator(Evaluator):
    model: AZModel
    observation_tensor_shape: tuple[int, int, int]
    nr_actions: int

    def evaluate(self, state):
        obs = self._get_observation(state)
        legals_mask = self._get_legals_mask(state)
        value = self.model.get_value(obs, legals_mask=legals_mask)
        value = (value[0], -value[0])
        return value

    def _get_legals_mask(self, state):
        legal_actions = state.legal_actions(state.current_player())
        mask = np.full((1, self.nr_actions), False)
        mask[0, legal_actions] = True
        return mask

    def prior(self, state):
        obs = self._get_observation(state)
        legals_mask = self._get_legals_mask(state)
        prior = self.model.get_prior(obs, legals_mask=legals_mask)
        legal_actions = state.legal_actions(state.current_player())
        actions_prior = list(zip(legal_actions, prior[0]))
        return actions_prior

    def _get_observation(self, state):
        return [state.observation_tensor()]

    @classmethod
    def load(
        cls,
        path: Path,
        observation_tensor_shape: tuple[int, int, int],
        nr_actions: int,
        checkpoint_nr: Optional[int] = None,
    ) -> AZEvaluator:
        model = load_az_model(path, checkpoint_nr)
        model.eval()
        return cls(model=model, observation_tensor_shape=observation_tensor_shape, nr_actions=nr_actions)


class MCTS(Player):
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(
        self,
        game: Game,
        uct_c: float,
        max_simulations: int,
        evaluator: Evaluator,
        solve=True,
        random_state=None,
        child_selection_fn=SearchNode.puct_value,
        dirichlet_noise=None,
        verbose=False,
        dont_return_chance_node=False,
        name=None,
    ):
        # Algorithm adapted from
        # https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mcts.py
        """Initializes a MCTS Search algorithm in the form of a bot.

        In multiplayer games, or non-zero-sum games, the players will play the
        greedy strategy.

        Args:
          game: A pyspiel.Game to play.
          uct_c: The exploration constant for UCT.
          max_simulations: How many iterations of MCTS to perform. Each simulation
            will result in one call to the evaluator. Memory usage should grow
            linearly with simulations * branching factor. How many nodes in the
            search tree should be evaluated. This is correlated with memory size and
            tree depth.
          evaluator: A `Evaluator` object to use to evaluate a leaf node.
          solve: Whether to back up solved states.
          random_state: An optional numpy RandomState to make it deterministic.
          child_selection_fn: A function to select the child in the descent phase.
            The default is UCT.
          dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
            the policy at the root. This is from the alpha-zero paper.
          verbose: Whether to print information about the search tree before
            returning the action. Useful for confirming the search is working
            sensibly.
          dont_return_chance_node: If true, do not stop expanding at chance nodes.
            Enabled for AlphaZero.

        Raises:
          ValueError: if the game type isn't supported.
        """
        # pyspiel.Bot.__init__(self)
        # Check that the game satisfies the conditions for this MCTS implemention.
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._game = game
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node
        self.name = name

    def __str__(self) -> str:
        name_str = "MCTS["
        if self.name is not None:
            name_str += f"{name_str}name={self.name}, "

        name_str += (
            f"uct_c={self.uct_c}, n_sims={self.max_simulations}, selection_fn={self._child_selection_fn.__name__}]"
        )
        return name_str

    def restart_at(self, state):
        pass

    def step_with_policy(self, state):
        """Returns bot's policy and action at given state."""
        t1 = time()
        root = self.mcts_search(state)

        best = root.best_child()

        if self.verbose:
            seconds = time() - t1
            print(
                "Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
                    root.explore_count, seconds, root.explore_count / seconds
                )
            )
            print("Root:")
            print(root.to_str(state))
            print("Children:")
            print(root.children_str(state))
            if best.children:
                chosen_state = state.clone()
                chosen_state.apply_action(best.action)
                print("Children of chosen:")
                print(best.children_str(chosen_state))

        mcts_action = best.action

        policy = [
            (action, (1.0 if action == mcts_action else 0.0)) for action in state.legal_actions(state.current_player())
        ]

        return policy, mcts_action

    def get_move(self, state) -> int:
        return self.step_with_policy(state)[1]

    def mcts_search(self, state):
        """A vanilla Monte-Carlo Tree Search algorithm.

        This algorithm searches the game tree from the given state.
        At the leaf, the evaluator is called if the game state is not terminal.
        A total of max_simulations states are explored.

        At every node, the algorithm chooses the action with the highest PUCT value,
        defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
        reward after the action, and N is the number of times the action was
        explored in this position. The input parameter c controls the balance
        between exploration and exploitation; higher values of c encourage
        exploration of under-explored nodes. Unseen actions are always explored
        first.

        At the end of the search, the chosen action is the action that has been
        explored most often. This is the action that is returned.

        This implementation supports sequential n-player games, with or without
        chance nodes. All players maximize their own reward and ignore the other
        players' rewards. This corresponds to max^n for n-player games. It is the
        norm for zero-sum games, but doesn't have any special handling for
        non-zero-sum games. It doesn't have any special handling for imperfect
        information games.

        The implementation also supports backing up solved states, i.e. MCTS-Solver.
        The implementation is general in that it is based on a max^n backup (each
        player greedily chooses their maximum among proven children values, or there
        exists one child whose proven value is game.max_utility()), so it will work
        for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
        draw games). Also chance nodes are considered proven only if all children
        have the same value.

        Some references:
        - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
          https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
        - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
          https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
        - Silver, AlphaGo Zero: Starting from scratch, 2017
          https://deepmind.com/blog/article/alphago-zero-starting-scratch
        - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
          https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

        Arguments:
          state: pyspiel.State object, state to search from

        Returns:
          The most visited move from the root node.
        """
        root = SearchNode(None, state.current_player(), 1)
        for nr_iterations in range(1, self.max_simulations + 1):
            self._mcts_iteration(root, state)
            if root.outcome is not None:
                break

        self.nr_iterations = nr_iterations

        return root

    def _mcts_iteration(self, root, state) -> None:
        visit_path, working_state = self._apply_tree_policy(root, state)
        if working_state.is_terminal():
            returns = working_state.returns()
            visit_path[-1].outcome = returns
            solved = self.solve
        else:
            returns = self.evaluator.evaluate(working_state)
            solved = False

        while visit_path:
            # For chance nodes, walk up the tree to find the decision-maker.
            decision_node_idx = -1
            while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
                decision_node_idx -= 1
            # Chance node targets are for the respective decision-maker.
            target_return = returns[visit_path[decision_node_idx].player]
            node = visit_path.pop()
            node.total_reward += target_return
            node.explore_count += 1

            if solved and node.children:
                player = node.children[0].player
                if player == pyspiel.PlayerId.CHANCE:
                    # Only back up chance nodes if all have the same outcome.
                    # An alternative would be to back up the weighted average of
                    # outcomes if all children are solved, but that is less clear.
                    outcome = node.children[0].outcome
                    if outcome is not None and all(np.array_equal(c.outcome, outcome) for c in node.children):
                        node.outcome = outcome
                    else:
                        solved = False
                else:
                    # If any have max utility (won?), or all children are solved,
                    # choose the one best for the player choosing.
                    best = None
                    all_solved = True
                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif best is None or child.outcome[player] > best.outcome[player]:
                            best = child
                    if best is not None and (all_solved or best.outcome[player] == self.max_utility):
                        node.outcome = best.outcome
                    else:
                        solved = False

    def _apply_tree_policy(self, root, state):
        """Applies the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.

        Args:
          root: The root node in the search tree.
          state: The state of the game at the root node.

        Returns:
          visit_path: A list of nodes descending from the root node to a leaf node.
          working_state: The state of the game at the leaf node.
        """
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while (not working_state.is_terminal() and current_node.explore_count > 0) or (
            working_state.is_chance_node() and self.dont_return_chance_node
        ):
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n) for (a, p), n in zip(legal_actions, noise)]
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [SearchNode(action, player, prior) for action, prior in legal_actions]

            if working_state.is_chance_node():
                # For chance nodes, rollout according to chance node's probability
                # distribution
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(c for c in current_node.children if c.action == action)
            else:
                # Otherwise choose node with largest UCT value
                chosen_child = max(
                    current_node.children,
                    key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                        c, current_node.explore_count, self.uct_c
                    ),
                )

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    @classmethod
    def load_pytorch_az(cls, path: str, max_simulations: Optional[int] = None) -> MCTS:
        path = Path(path)
        config_dict = load_json(path / "config.json")
        game = load_game(config_dict["game"])
        observation_tensor_shape = tuple(game.observation_tensor_shape())
        az_evaluator = AZEvaluator.load(path, observation_tensor_shape, nr_actions=game.num_distinct_actions())
        uct_c = config_dict["uct_c"]

        if max_simulations is None:
            max_simulations: int = config_dict["max_simulations"]

        return cls(
            game=game,
            uct_c=uct_c,
            max_simulations=max_simulations,
            evaluator=az_evaluator,
            child_selection_fn=SearchNode.puct_value,
            random_state=np.random.RandomState(seed=42),
        )

    @classmethod
    def load_tensorflow_az(
        cls,
        path: str | Path,
        max_simulations: Optional[int] = None,
        uct_c: Optional[float] = None,
        checkpoint_nr: Optional[int] = None,
        name: Optional[str] = None,
    ) -> MCTS:
        game, uct_c, checkpoint_nr, max_simulations, evaluator = cls._get_tensorflow_info(path, uct_c, checkpoint_nr)
        if name is None:
            name = f"{str(path)}_checkpoint_nr_{checkpoint_nr}"

        return cls(
            game=game,
            max_simulations=max_simulations,
            uct_c=uct_c,
            evaluator=evaluator,
            child_selection_fn=SearchNode.puct_value,
            name=name,
            random_state=np.random.RandomState(seed=42),
        )

    @staticmethod
    def _get_tensorflow_info(
        path: str | Path,
        uct_c: Optional[float] = None,
        checkpoint_nr: Optional[int] = None,
        max_simulations: Optional[int] = None,
    ) -> tuple[Game, float, int, int, Evaluator]:
        path = Path(path)
        config_dict = load_json(path / "config.json")
        game = PlayerNrGameWrapper.from_game_name(config_dict["game"])
        observation_tensor_shape = tuple(config_dict["observation_shape"])

        if checkpoint_nr is None:
            checkpoint_nr = _get_last_checkpoint(path)

        model = TFModel.from_checkpoint(str(path / f"checkpoint-{checkpoint_nr}"))
        evaluator = AZEvaluator(model, observation_tensor_shape, config_dict["output_size"])
        if uct_c is None:
            uct_c = config_dict["uct_c"]
        if max_simulations is None:
            max_simulations = config_dict["max_simulations"]

        return game, uct_c, checkpoint_nr, max_simulations, evaluator


class TimeMCTS(MCTS):
    def __init__(
        self,
        game: Game,
        time_budget: float,
        uct_c: float,
        evaluator: Evaluator,
        solve=True,
        random_state=None,
        child_selection_fn=SearchNode.puct_value,
        dirichlet_noise=None,
        verbose=False,
        dont_return_chance_node=False,
        name=None,
    ):
        super().__init__(
            game=game,
            uct_c=uct_c,
            max_simulations=float("inf"),
            evaluator=evaluator,
            solve=solve,
            random_state=random_state,
            child_selection_fn=child_selection_fn,
            dirichlet_noise=dirichlet_noise,
            verbose=verbose,
            dont_return_chance_node=dont_return_chance_node,
            name=name,
        )
        self.time_budget = time_budget

    def mcts_search(self, state):
        self.start_time = time()
        root = SearchNode(None, state.current_player(), 1)
        for nr_iterations in count(start=1):
            if self._is_out_of_time():
                break

            self._mcts_iteration(root, state)
            if root.outcome is not None:
                break

        self.nr_iterations = nr_iterations

        return root

    def _is_out_of_time(self):
        current_time = time()
        time_passed = current_time - self.start_time
        return time_passed > self.time_budget

    def __str__(self) -> str:
        name_str = "TimeMCTS["
        if self.name is not None:
            name_str += f"{name_str}name={self.name}, "

        name_str += (
            f"uct_c={self.uct_c}, time_budget={self.time_budget}, selection_fn={self._child_selection_fn.__name__}]"
        )
        return name_str

    @classmethod
    def load_pytorch_az(cls, path: str, time_budget: float, checkpoint_nr: Optional[int] = None) -> TimeMCTS:
        path = Path(path)
        config_dict = load_json(path / "config.json")
        game = load_game(config_dict["game"])
        observation_tensor_shape = tuple(game.observation_tensor_shape())
        az_evaluator = AZEvaluator.load(path, observation_tensor_shape, nr_actions=game.num_distinct_actions())
        uct_c = config_dict["uct_c"]

        return cls(
            game=game,
            time_budget=time_budget,
            uct_c=uct_c,
            evaluator=az_evaluator,
            child_selection_fn=SearchNode.puct_value,
            name=f"{str(path)}_checkpoint_nr_{checkpoint_nr}",
            random_state=np.random.RandomState(seed=42),
        )

    @classmethod
    def load_tensorflow_az(
        cls,
        path: str | Path,
        time_budget: float,
        uct_c: Optional[float] = None,
        checkpoint_nr: Optional[int] = None,
        name: Optional[str] = None,
    ) -> TimeMCTS:
        game, uct_c, checkpoint_nr, _, evaluator = cls._get_tensorflow_info(path, uct_c, checkpoint_nr)

        if name is None:
            name = f"{str(path)}_checkpoint_nr_{checkpoint_nr}"

        return cls(
            game=game,
            time_budget=time_budget,
            uct_c=uct_c,
            evaluator=evaluator,
            child_selection_fn=SearchNode.puct_value,
            name=name,
            random_state=np.random.RandomState(seed=42),
        )


@dataclass
class ConfigurationPlayer(Player):
    # config predictor takes state and predicts the best exploration constant for the MCTS player
    config_predictor: Callable
    mcts: MCTS
    c_list: list[float]
    name: Optional[str] = None

    def __post_init__(self):
        self.config_predictor.eval()

    def get_move(self, state) -> int:
        obs = state.observation_tensor()
        features = np.array(obs)
        feature_shape = tuple(state.game.observation_tensor_shape())
        features = features.reshape(feature_shape)
        obs = torch.tensor([features], dtype=torch.float32, device=DEVICE)
        uct_c_index, _ = self.config_predictor.actor(dict(obs=obs))
        uct_c_index = uct_c_index.argmax().item()
        uct_c = self.c_list[uct_c_index]
        self.mcts.uct_c = uct_c
        return self.mcts.get_move(state)

    @property
    def nr_iterations(self) -> int:
        return self.mcts.nr_iterations

    @classmethod
    def from_path(cls, path: str | Path, mcts: MCTS, c_list: list[float]) -> ConfigurationPlayer:
        config_predictor = torch.load(path, map_location=DEVICE)
        return cls(config_predictor, mcts, c_list)

    def __str__(self) -> str:
        if self.name is not None:
            return f"ConfigurationMCTS[name={self.name}]"

        return f"ConfigurationMCTS[uct_c=-1]"


def _get_last_checkpoint(path: Path | str) -> int:
    path = Path(path)

    for checkpoint_nr in count(start=1):
        checkpoint_path = path / f"checkpoint-{checkpoint_nr}.meta"

        if not checkpoint_path.exists():
            return checkpoint_nr - 1
