import numba
from numba import prange
import numpy as np

def floyd_warshall(n: int, ei):
    D = np.full((n, n), 127, dtype=np.uint8)
    ei_tuple = (ei[0, :], ei[1, :])
    D[ei_tuple] = 1
    D[ei_tuple[::-1]] = 1
    return _floyd_warshall_wrapper(n, D)

@numba.njit()
def _floyd_warshall_wrapper(n, D):
    np.fill_diagonal(D, 0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, j] > D[i, k] + D[k, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D

def graph_dist_transform(D):
    D = np.square(D - 1) / 10
    return np.minimum(D, 1)
