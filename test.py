import numpy as np
import skimage

from utils import *

N_SEGMENTS = 3

if __name__ == '__main__':
    # reading in an image
    #img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    #pieces = generate_puzzle(img, n_segments=N_SEGMENTS)

    test_fitness_matrix_pair1 = (np.array([
        [10, 10],
        [10, 10],
        [10, 1]
    ]), np.array([
        [1, 10, 10],
        [10, 10, 1]
    ]))

    generate_offspring(None, test_fitness_matrix_pair1, None, None,
        2, N_SEGMENTS)
