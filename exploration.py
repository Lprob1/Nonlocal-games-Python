from pprint import pprint
from itertools import product
from typing import Any
from typing import Callable
from typing import Iterable, Tuple, Hashable, List
import itertools

def pprint_input_output(strat: tuple[tuple[Any], tuple[Any]]):
    """
    Docstring for pprint_input_output
    
    :param strat: A pair of input-output
    :type strat: tuple[tuple[Any], tuple[Any]]
    """
    
    inp = strat[0]
    out = strat[1]
    print(f"{inp} -> {out}")

X = [0,1]
A = [0,1] #tried it with 0,1,2, quite slower
n_players = 2
#winning predicate for the CHSH game
def W(x,y,a,b):
    tmp = x * y
    tmp2 = (a + b) % 2
    if tmp == tmp2:
        return 1
    else:
        return 0


def strategy_tuple_to_dict(strategy_tuple):
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

view_for_one_player = list(product(X, A)) # list of possible input-output pairs for one bit
#I want a list that looks like this:

#at last, that's it.
X_t = Hashable
A_t = Hashable
Strategy = Tuple[Tuple[X_t, A_t], ...]

def all_strategies(X: Iterable[X_t], A: Iterable[A_t]) -> List[Strategy]:
    X = tuple(X)
    A = tuple(A)
    return [tuple(zip(X, outputs)) for outputs in product(A, repeat=len(X))]

strats = all_strategies(X, A)
n_players_strats = list(product(strats, repeat=n_players))

dict_strats = []
for strat in n_players_strats:
    # how many tuples will there be? -> n_players , each with |X| tuples
    # strat is smtg like this: (((0, 0), (1, 0)), ((0, 0), (1, 0)))
    #this is behavior data. I must generate all possible questions, then find the behavior
    dict_strat = strategy_tuple_to_dict(strat)
    dict_strats.append(dict_strat)

    
for s in dict_strats:
    print(s)

def calculate_chsh_exclusion(strategy: dict, W: Callable):
    questions = strategy.keys()
    exclusion_set = set()
    for q in questions:
        x = q[0]
        y = q[1]
        ans = strategy[q]
        a = ans[0]
        b = ans[1]
        winning = W(x, y, a, b)
        if winning == 0:
            exclusion_set.add((x, y))
    return exclusion_set

all_exclusion_sets = [calculate_chsh_exclusion(s, W) for s in dict_strats]
# print(all_exclusion_sets)
