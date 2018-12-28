import numpy as np

from utils import *

N_SEGMENTS = 5

if __name__ == '__main__':

    # testing get_ind_stats() function
    '''fitness_matrix_pair = (np.array([
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
    ]), np.zeros((N_SEGMENTS, N_SEGMENTS), dtype=int))

    cluster_matrix, cluster_fitnesses, match_orientations = get_ind_stats(
        ind, 2, N_SEGMENTS, fitness_matrix_pair=fitness_matrix_pair)

    print('\nPiece indices:')
    print(ind[1])
    print('\nRotations:')
    print(ind[2])
    print('\nCluster matrix:')
    print(cluster_matrix)
    print('\nCluster fitnesses:')
    print(cluster_fitnesses)
    print('\nMatch-orientation array:')
    print(match_orientations)
    print()'''
