from pprint import pprint
from itertools import product
from typing import Any
from typing import Callable
from typing import Iterable, Tuple, Hashable, List

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


view_for_one_player = list(product(X, A)) # list of possible input-output pairs for one bit
#I want a list that looks like this:
"""
[
    X = [0, 1]
    A = [0, 1]
    ((0, 0), (1, 0))
    ((0, 0), (1, 1))
    ((0, 1), (1, 0))
    ((0, 1), (1, 1))
    X = [0, 1, 2]
    A = [0, 1]
    ((0, 0), (1, 0), (2, 0))
    ((0, 0), (1, 0), (2, 1))
    ((0, 0), (1, 1), (2, 0))
    ((0, 0), (1, 1), (2, 1))
    ((0, 1), (1, 0), (2, 0))
    ((0, 1), (1, 0), (2, 1))
    ((0, 1), (1, 1), (2, 0))
    ((0, 1), (1, 1), (2, 1))
    
]

the length of the subset is the number of elements in the input set (len(X))
There will be len(A)^len(X) subsets -> len(A) * lan(A) * ... * len(A)
Each of these subsets is a strategy.
Cartesian product of A, repeated len(X) times -> zipped with X
"""

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
print(len(n_players_strats))
print(n_players_strats)

#now, this looks like this:
"""
[
    ((strategy for player 1), (strategy for player 2))
]
and strategies look liks ((),()), with one sub-tuple for each input in X
The sub tuples are input-output pairs
"""
#let's turn the strategies in dicts, just to make it a bit more reader friendly
"""
I would like each strategy to look like this:
strategy = {
    (0, 0) -> (output)
    (0, 1) -> (output)
    (1, 0) -> (output)
    (1, 1) -> (output)
}
So that we can see exactly how the strategy behaves
"""
dict_strats = []
for strat in n_players_strats:
    # how many tuples will there be? -> n_players , each with |X| tuples
    # strat is smtg like this: (((0, 0), (1, 0)), ((0, 0), (1, 0)))
    #this is behavior data. I must generate all possible questions, then find the behavior
    dict_strat = {}
# FINISH THIS   

# for strat in n_players_strats:
#     #how many tuples will there be? -> n_players , each with |X| tuples
#     dict_strat = {}
#     for i, x in enumerate(X):
#         question_answer = []
#         for player_tup in strat:
#             question_answer.append(player_tup[i])
#         #now, question_answer = [(0, 0), (1, 0)], map that
#         # to (0, 1): (0, 0)
#         question = []
#         answer = []
#         for j in range(len(X)):
#             question.append(question_answer[j][0])
#             answer.append(question_answer[j][1])
#         dict_strat[tuple(question)] = tuple(answer)
        
#     dict_strats.append(dict_strat)
    
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
print(all_exclusion_sets)
