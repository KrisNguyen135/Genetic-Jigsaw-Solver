import numpy as np
import skimage

from utils import *

N_SEGMENTS = 3

if __name__ == '__main__':
    # reading in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    #pieces = generate_puzzle(img, n_segments=N_SEGMENTS)
    # to be uncommented when testing unshuffled image
    pieces = generate_puzzle(
        img, n_segments=N_SEGMENTS, shuffle=False)

    # calculating threshold for finding good matches
    #threshold = generate_threshold(pieces)
    #print('Threshold:', threshold)

    # generating the piece edges, clock-wise
    piece_edges = np.array([np.array([
        piece[0, :],
        piece[:, -1],
        piece[-1, :],
        piece[:, 0]
    ]) for piece in pieces])
    '''print('Original piece-edge array:')
    print(piece_edges)
    print(piece_edges.shape)'''

    # testing of unshuffled individual
    '''unshuffled_ind_v1 = (
        piece_edges.reshape((N_SEGMENTS, N_SEGMENTS, 4, -1)),
        np.arange(N_SEGMENTS * N_SEGMENTS).reshape((N_SEGMENTS, N_SEGMENTS)),
        np.zeros((N_SEGMENTS * N_SEGMENTS), dtype=int)
    )

    print('Fitness version 1:')
    print(get_fitness(unshuffled_ind_v1, N_SEGMENTS))

    unshuffled_ind_v2 = (
        np.arange(N_SEGMENTS * N_SEGMENTS).reshape((N_SEGMENTS, N_SEGMENTS)),
        np.zeros((N_SEGMENTS * N_SEGMENTS), dtype=int)
    )

    print('Fitness version 2:')
    print(get_fitness_v2(piece_edges, unshuffled_ind_v2, N_SEGMENTS))'''

    # testing of population-generating functions
    '''pop = generate_init_pop(piece_edges, N_SEGMENTS)
    ind = pop[0]
    print('Individual piece indices:')
    print(ind[1])
    print('Individual orientations:')
    print(ind[2])
    print('Individual fitness:')
    #print('Version 1 piece-edge array:')
    #print(ind[0])
    print(get_fitness(ind, N_SEGMENTS))'''

    '''pop = generate_init_pop_v2(N_SEGMENTS)
    ind = pop[0]
    print('Individual piece indices:')
    print(ind[0])
    #print('Invidual orientations:')
    #print(ind[1])
    print('Individual fitness:')
    print(get_fitness_v2(piece_edges, ind, N_SEGMENTS))'''

    # testing visualization functions
    '''pop = generate_init_pop(piece_edges, N_SEGMENTS)
    ind = pop[0]
    visualize(pieces, ind, N_SEGMENTS)'''

    '''pop = generate_init_pop_v2(N_SEGMENTS)
    ind = pop[0]
    visualize_v2(pieces, ind, N_SEGMENTS)'''
