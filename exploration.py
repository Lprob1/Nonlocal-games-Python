from pprint import pprint
from itertools import product
from itertools import count
from typing import Any
from typing import Callable
from typing import Iterable, Tuple, Hashable, List
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from pysat.formula import CNF
from pysat.solvers import Solver

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

for strat in dict_strats:
    print(strat)

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
frozen_all_exclusion_sets = [frozenset(s) for s in all_exclusion_sets]
#this must be turned into frozensets (unmutable)
unique_exclusion_sets = set(frozen_all_exclusion_sets)
unique_exclusion_sets_list = list(unique_exclusion_sets)

# each strategy will be one number, that's how well label the vertices
# each exclusion set will have a number (color), that's how we'll color the graph
#create a dict of strategies: exclusion set mapping to color the graph
#iterate over this and make a graph with coloured nodes
#for every strategy, for every question, make a table and compute the answers of players
# for every t in range (n_players), apply the procedure to make the links between the vertices.

#create a mapping exclusion set -> color
set_coloring = {}
for i, fset in enumerate(unique_exclusion_sets_list):
    set_coloring[fset] = i

#create the dict of strategies -> exclusion set

strategies_to_exclusion_set_mapping = {}
#note that since dict is unhashable, we can't use that as an index, so we have to use numbers to index strategies
for i, fset in enumerate(frozen_all_exclusion_sets):
    strategies_to_exclusion_set_mapping[i] = set_coloring[fset]

# print(strategies_to_exclusion_set_mapping)

#create the graph
G = nx.MultiGraph()
for s, c in strategies_to_exclusion_set_mapping.items():
    G.add_node(s, col= c)
    

#create the table
n_strategies = len(n_players_strats)
all_questions = list(product(X, repeat=n_players))
n_questions = len(all_questions)
len_answers = n_players # length of answers, just for clarity
#so, the table is going to be a 3D array: rows: strategies, cols: questions, depth: player response
answers_table = np.ndarray(shape=(n_strategies, n_questions, len_answers), dtype=int) #we assume int for now, any data could be represented with some int
for s_number in range(n_strategies):
    for q_number in range(n_questions):
        #compute s(x)
        strategy = dict_strats[s_number]
        question = all_questions[q_number]
        answer = strategy[question]
        for t in range(len(answer)):
            answers_table[s_number, q_number, t] = answer[t]
#the shape of the answer table seems correct

def satisfies_hiding_criterion(a_i, a_j, x, t, W):
    if W(x, a_i) == 1 and W(x, a_j) == 1: #if they both win
        #they normally have the same length
        for q in range(len(a_i)):
            if q == t:
                continue
            else:
                if a_i[q] != a_j[q]:
                    return False
    else:
        return False
    return True


#now, fill the graph edges
#right now, we compute everything in the table twice, it's not optimal
# becaues we compute s_i, s_j in the predicate
def Winning_predicate(x, a):
    """
    Docstring for Winning_predicate
    
    :param x: question
    :param a: answer
    """
    tmp1 = x[0]*x[1]
    tmp2 = (a[0] + a[1]) % 2
    if tmp1 == tmp2:
        return 1
    else:
        return 0
    
for t in range(1): #just one for now, but we 
    for q_number in range(n_questions):
        i = 0
        x = all_questions[q_number]
        while i < n_strategies:
            for j in range(i+1, n_strategies):
                #compare s_i(x) and s_j(x)
                a_i = answers_table[i, q_number, :]
                a_j = answers_table[j, q_number, :]
                if satisfies_hiding_criterion(a_i, a_j, x, t, Winning_predicate):
                    #add edge
                    G.add_edge(i, j, label=q_number)
            i += 1 #don't forget that...

# print("num edges:", G.number_of_edges())
# print("num labels:", len(nx.get_edge_attributes(G, "label")))

#now, turn this into a SAT problem


#this works to display G with colors, we can add edges later
#let's just try to display G with a coloring
cols = set(nx.get_node_attributes(G, "col").values())
mapping = dict(zip(sorted(cols),count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['col']] for n in nodes]
edge_labels = nx.get_edge_attributes(G, "label")


pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2, connectionstyle="arc3,rad=0.15")
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
el = nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=8
)
plt.colorbar(nc)
plt.axis('off')
plt.show()