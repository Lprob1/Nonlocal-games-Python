from itertools import product
from itertools import count
from typing import Any
from typing import Callable
from typing import Iterable, Tuple, Hashable, List
import itertools

#some types
X_t = Hashable
A_t = Hashable
Strategy = Tuple[Tuple[X_t, A_t], ...]

class Game:
    def __init__(self, n: int, X: Iterable, A: Iterable, support: Iterable, W: Callable, pi: Callable, consider_only_optimal=True):
        self.n = n
        self.X = X
        self.A = A
        self.support = support
        self.W = W
        self.pi = pi
        self.consider_only_optimal = consider_only_optimal
        
        #derived properties
        self.n_players_strategies = None
        self.exclusion_sets = None
        #derive the propertiess
        self.compute_n_players_strategies()
        self.calculate_exclusion_sets()

    def is_valid_question(self, x: Tuple) -> bool:
        """
        Enforces if x is a valid question
        """
        raise NotImplementedError("haha")
    
    def _strategy_tuple_to_dict(self, strategy_tuple):
        """
        Converts a deterministic strategy tuple into a dictionary:
        (x1, ..., xn) -> (a1, ..., an)
        """
        # Build per-player input -> output maps
        player_maps = []
        for player in strategy_tuple:
            player_maps.append(dict(player))

        # Assume all players share the same input alphabet
        inputs = list(player_maps[0].keys())

        strategy = {}
        for joint_input in itertools.product(inputs, repeat=len(player_maps)):
            joint_output = tuple(
                player_maps[i][joint_input[i]]
                for i in range(len(player_maps))
            )
            strategy[joint_input] = joint_output

        return strategy
    
    def compute_deterministic_strategies(self):
        X = tuple(self.X)
        A = tuple(self.A)
        return [tuple(zip(X, outputs)) for outputs in product(A, repeat=len(X))]

    def compute_n_players_strategies(self):
        n_players_strategies= list(product(self.compute_deterministic_strategies, repeat=self.n))
        tmp = []
        for strat in n_players_strategies:
            dict_strat = self._strategy_tuple_to_dict(strat)
            tmp.append(dict_strat)
        self.n_players_strategies = tmp

    def _calculate_strategy_exclusion_set(self, strategy: dict, W: Callable) -> set:
        questions = strategy.keys()
        exclusion_set = set()
        for q in questions:
            a = strategy[q]
            if W(q, a) == 1:
                exclusion_set.add((q, a))
        return exclusion_set
    
    def calculate_exclusion_sets(self):
        self.exclusion_sets = [self._calculate_strategy_exclusion_set(s, self.W) for s in self.n_players_strategies]
        

class NonLocalGameBitCommitment:
    def __init__(self, game: Game):
        self.game = game
    
    