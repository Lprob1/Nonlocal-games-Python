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

#magic_square values
X = [0,1,2]
#for magic square, we have to find a way to generate the matrices proceduraly
A = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
A = ["000", "001", "010", "011", "100", "101", "110", "111"]

#somehow, in the way that we create the strategy, we end up with one of them having NO exclusion set. This is impossible, let's figure out how?
#I think it has to do with the fact that I make a product of these triples, and some of them actually encode matrices that are not possible. We need to procedurally generate matrices, 
# then generate the strategies dict from this
# since the hiding criterion cares only about when they are winning, we can always assume that the ? in the matrix will always be a one. Ie, in the strategy,
# the players will always answer 1 in that spot, and its functionally equivalent to the strategy with a 0, since they both loose either way.
all_strategy_matrices = []



n_players = 2

def winning_predicate(x, a):
    row = a[0]
    col = a[1]
    xor_row = sum([int(x) for x in row]) % 2
    xor_col = sum([int(x) for x in col]) % 2
    i = x[0]
    j = x[1]
    row_i_col_j = (row[i] == col[j])
    if xor_row == 0 and xor_col == 1 and row_i_col_j:
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

# for strat in dict_strats:
#     print(strat)

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

def calculate_exclusion_sets(strategy: dict, W: Callable):
    questions = strategy.keys()
    exclusion_set = set()
    for q in questions:
        ans = strategy[q]
        winning = W(q, ans)
        if winning == 0:
            exclusion_set.add(q)
    return exclusion_set

all_exclusion_sets = [calculate_exclusion_sets(s, winning_predicate) for s in dict_strats]
frozen_all_exclusion_sets = [frozenset(s) for s in all_exclusion_sets]
#this must be turned into frozensets (unmutable)
unique_exclusion_sets = set(frozen_all_exclusion_sets)
unique_exclusion_sets_list = list(unique_exclusion_sets)

#we are going to remove the strategies that are not optimal
# for now, we assume the distribution is uniform, so keep just the smallest exclusion sets
min_set_size = min([len(s) for s in all_exclusion_sets])
optimal_strategies = []
optimal_sets = []
for i, strat in enumerate(dict_strats):
    if len(all_exclusion_sets[i]) == min_set_size:
        optimal_strategies.append(strat)
        optimal_sets.append(frozen_all_exclusion_sets[i])
#just relabel with the newly created variables to not rewrite too much code
dict_strats = optimal_strategies
frozen_all_exclusion_sets = optimal_sets
# for s in dict_strats:
#     print(s)
#redo the unique ones
unique_exclusion_sets = set(frozen_all_exclusion_sets)
unique_exclusion_sets_list = list(unique_exclusion_sets)
print(unique_exclusion_sets_list) #for some reason, this is a ilist with only an empty set...

#create a mapping exclusion set -> color
set_coloring = {}
for i, fset in enumerate(unique_exclusion_sets_list):
    set_coloring[fset] = i

#create the dict of strategies -> exclusion set

strategies_to_exclusion_set_mapping = {}
#note that since dict is unhashable, we can't use that as an index, so we have to use numbers to index strategies
for i, fset in enumerate(frozen_all_exclusion_sets):
    strategies_to_exclusion_set_mapping[i] = set_coloring[fset]

#create the graph
G = nx.MultiGraph()
for s, c in strategies_to_exclusion_set_mapping.items():
    G.add_node(s, col= c)
    

#create the table
n_strategies = len(dict_strats)
all_questions = list(product(X, repeat=n_players))
n_questions = len(all_questions)
len_answers = n_players # length of answers, just for clarity
#so, the table is going to be a 3D array: rows: strategies, cols: questions, depth: player response
answers_table = np.ndarray(shape=(n_strategies, n_questions, len_answers), dtype=Any) #was previously int, but for magic square its more convenient to encode rows of matrices as strings
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
    if W(x, a_i) == 1 and W(x, a_j) == 1: #if they both win // 
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


for t in range(1): #just one for now, but we 
    for q_number in range(n_questions):
        i = 0
        x = all_questions[q_number]
        while i < n_strategies:
            for j in range(i+1, n_strategies):
                #compare s_i(x) and s_j(x)
                a_i = answers_table[i, q_number, :]
                a_j = answers_table[j, q_number, :]
                if satisfies_hiding_criterion(a_i, a_j, x, t, winning_predicate):
                    #add edge
                    G.add_edge(i, j, label=q_number)
            i += 1 #don't forget that...

clauses = []

nodes = G.nodes() #normally, just the list of strategies
# print(nodes)
for node in nodes:
    neighbours = G[node] #gets all the neighbours
    # print(neighbours)
    #what we want: find S(v, t): neighbours of vertex v with edge type t
    #we have to iterate over all edge types
    
    for tau in range(n_questions): #edge type t, t is already taken for number of players
        S_vt = []
        for neighbour, edges in neighbours.items():
            #now, iterate over the edges, check if they are equal to tau
            #edges is a dict
            for edge_id, edge in edges.items():
                if edge["label"] == tau:
                    S_vt.append(neighbour)
        #create OR clause with COLORS of nodes of S_vt
        S_vt_colors = [G.nodes[s]["col"] for s in S_vt]
        #add the node itself
        S_vt_colors.append(G.nodes[node]["col"])
        OR_clause_1 = [x+1 for x in S_vt_colors]
        OR_clause_2 = [-x-1 for x in S_vt_colors] #doesnt want x_0, so have to convert back after
        clauses.append(OR_clause_1)
        clauses.append(OR_clause_2)
    #create OR clauses with nodes with COLORS of S_vt
           
#plug in the solver
# print(clauses)
cnf = CNF(from_clauses=clauses)
with Solver(bootstrap_with=cnf) as solver:
    # 1.1 call the solver for this formula:
    print('formula is', f'{"s" if solver.solve() else "uns"}atisfiable')

    # 1.2 the formula is satisfiable and so has a model:
    print('and the model is:', solver.get_model())

    # 2.2 the formula is unsatisfiable,
    # i.e. an unsatisfiable core can be extracted:
    print('and the unsatisfiable core is:', solver.get_core())

#this works to display G with colors, we can add edges later
#let's just try to display G with a coloring
cols = set(nx.get_node_attributes(G, "col").values())
mapping = dict(zip(sorted(cols),count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['col']] for n in nodes]


pos = nx.spring_layout(G)
edge_labels = {}
for u, v, k, d in G.edges(keys=True, data=True):
    edge_labels.setdefault((u, v), []).append(str(d["label"]))

edge_labels = {e: ",".join(lbls) for e, lbls in edge_labels.items()}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2, ) #connectionstyle="arc3,rad=0.15"
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
el = nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=8
)
lb = nx.draw_networkx_labels(
    G, pos,
    labels={n: str(n) for n in G.nodes()},
    font_size=9,
    font_color="white"
)
plt.colorbar(nc)
plt.axis('off')
plt.show()