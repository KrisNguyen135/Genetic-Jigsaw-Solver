import numpy as np

from utils import *

N_SEGMENTS = 5

if __name__ == '__main__':

    test_fitness_matrix_pair1 = (np.array([
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

    parent1 = (None, np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ]), np.zeros((N_SEGMENTS, N_SEGMENTS)))

    generate_offspring(parent1, test_fitness_matrix_pair1, None, None,
        2, N_SEGMENTS)
