import numpy as np

from utils import *

N_SEGMENTS = 3

if __name__ == '__main__':

    # testing get_ind_stats() function
    '''N_SEGMENTS = 5
    fitness_matrix_pair = (np.array([
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [10, 10, 0, 1],
        [10, 10, 1, 10],
        [10, 10, 10, 10]
    ]), np.array([
        [10, 10, 10, 10, 10],
        [1, 10, 10, 10, 10],
        [0, 10, 1, 1, 10],
        [10, 10, 10, 10, 10]
    ]))

    ind = (None, np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ]), np.zeros((N_SEGMENTS * N_SEGMENTS,), dtype=int))

    cluster_matrix, cluster_id_set, cluster_fitnesses, cluster_to_piece_set,\
        match_orientations = get_ind_stats(
            ind, 2, N_SEGMENTS, fitness_matrix_pair = fitness_matrix_pair
        )

    print('\nPiece indices:')
    print(ind[1])
    print('\nRotations:')
    print(ind[2])
    print('\nCluster matrix:')
    print(cluster_matrix)
    print('\nCluster id set:')
    print(cluster_id_set)
    print('\nCluster fitnesses:')
    print(cluster_fitnesses)
    print('\n Cluster to piece')
    print(cluster_to_piece_set)
    print('\nMatch-orientation array:')
    print(match_orientations)
    print()'''


    # testing generate_offspring() function
    parent1 = (None, np.array([
        [4, 3, 0],
        [5, 6, 1],
        [2, 8, 7]
    ]), np.array([1, 1, 0, 1, 0, 0, 0, 3, 0]))

    # non-conflicting case
    parent2 = (None, np.array([
        [1, 0, 6],
        [4, 2, 5],
        [7, 3, 8]
    ]), np.array([2, 1, 3, 0, 0, 3, 3, 0, 2]))

    # mergeable case
    '''parent2 = (None, np.array([
        [3, 4, 6],
        [2, 5, 8],
        [1, 7, 0]
    ]), np.array([0, 3, 3, 2, 0, 1, 0, 1, 0]))'''

    # conflicting case
    '''parent2 = (None, np.array([
        [3, 8, 7],
        [0, 2, 6],
        [4, 1, 5]
    ]), np.array([0, 1, 2, 3, 0, 0, 0, 0, 3]))'''

    # mergeable multiple common pieces case
    '''parent2 = (None, np.array([
        [1, 2, 6],
        [0, 8, 5],
        [3, 7, 4]
    ]), np.array([0, 3, 2, 0, 0, 1, 2, 2, 0]))'''

    # large cluster case
    parent2 = (None, np.array([
        [2, 7, 4],
        [6, 8, 5],
        [3, 1, 0]
    ]), np.array([0, 0, 2, 3, 1, 1, 0, 1, 1]))

    generate_offspring(parent1, parent2, 2, N_SEGMENTS)
