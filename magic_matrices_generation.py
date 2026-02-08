from itertools import product
from typing import List, Tuple, Dict

N = 3

#for each i and j, there are 32 matrices...
#we can reduce that to 16 by setting that i,j to be 1 all the time instead of 1 or 0
#we also actually need to make the players adapt, in the sense that if they are asked this row or col,
# they will respond with the value that makes their row or column xor to the proper value. Then,
# the condition that will fail the winning condition is r[i] == c[j]
# think about it...

def row_xor(M: List[List[int]], r: int) -> int:
    x = 0
    for c in range(N):
        x ^= M[r][c]
    return x

def col_xor(M: List[List[int]], c: int) -> int:
    x = 0
    for r in range(N):
        x ^= M[r][c]
    return x

def bits_to_matrix(bits: Tuple[int, ...]) -> List[List[int]]:
    # bits length 9, row-major
    return [list(bits[r*N:(r+1)*N]) for r in range(N)]

def matrix_to_str(M: List[List[int]]) -> str:
    return "\n".join(" ".join(str(x) for x in row) for row in M)

def perfect_matrices() -> List[List[List[int]]]:
    out = []
    for bits in product([0, 1], repeat=N*N):
        M = bits_to_matrix(bits)
        if all(row_xor(M, r) == 0 for r in range(N)) and all(col_xor(M, c) == 1 for c in range(N)):
            out.append(M)
    return out

def weak_matrices_for_cell(i: int, j: int) -> List[List[List[int]]]:
    out = []
    for bits in product([0, 1], repeat=N*N):
        M = bits_to_matrix(bits)

        # constraints on all other rows/cols
        ok_others = True
        for r in range(N):
            if r != i and row_xor(M, r) != 0:
                ok_others = False
                break
        if not ok_others:
            continue

        for c in range(N):
            if c != j and col_xor(M, c) != 1:
                ok_others = False
                break
        if not ok_others:
            continue

        # forbidden combination on the selected row/col
        if row_xor(M, i) == 0 and col_xor(M, j) == 1:
            continue

        out.append(M)
    return out

def main(print_matrices: bool = False) -> None:
    perf = perfect_matrices()
    print(f"Perfect matrices (all rows XOR=0, all cols XOR=1): {len(perf)}")

    results: Dict[Tuple[int, int], List[List[List[int]]]] = {}
    for i in range(N):
        for j in range(N):
            mats = weak_matrices_for_cell(i, j)
            results[(i, j)] = mats
            print(f"Weak matrices for cell (i={i}, j={j}): {len(mats)}")
            if print_matrices:
                for k, M in enumerate(mats, 1):
                    print(f"\n(i={i}, j={j}) matrix #{k}")
                    print(matrix_to_str(M))

if __name__ == "__main__":
    # Set to True if you want to print every matrix.
    main(print_matrices=True)