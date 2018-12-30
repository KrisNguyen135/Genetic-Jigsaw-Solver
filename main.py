import numpy as np
import skimage

from utils import *

N_SEGMENTS = 3

if __name__ == '__main__':
    # reading in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    pieces = generate_puzzle(img, n_segments=N_SEGMENTS)
    # to be uncommented when testing unshuffled image
    #pieces = generate_puzzle(img, n_segments=N_SEGMENTS, shuffle=False)


    '''print('Individual pieces:')
    for piece in pieces:
        print(piece)
    print(pieces.shape)
    print('*' * 50)'''


    threshold = generate_threshold(pieces)
    print('Threshold:', threshold)


    # generating the piece edges, clock-wise
    piece_edges =  np.array([np.array([
        piece[0, :],
        piece[:, -1],
        #np.flip(piece[-1, :]),
        piece[-1, :], # easier to calculate differences
        #np.flip(piece[:, 0])
        piece[:, 0] # easier to calculate differences
    ]) for piece in pieces])

    '''print('Piece edges:')
    for piece in piece_edges:
        print(piece)
    print(piece_edges.shape)
    print('*' * 50)'''


    # testing unshuffled image
    # switch the code that generates the puzzle when uncomment this part
    '''ind = (
        piece_edges,
        np.arange(N_SEGMENTS ** 2).reshape((N_SEGMENTS, N_SEGMENTS)),
        np.zeros((N_SEGMENTS * N_SEGMENTS,), dtype=int)
    )

    print('Fitness:', get_fitness(ind, N_SEGMENTS))
    visualize(pieces, ind, N_SEGMENTS)'''


    # generating the initial random population
    #init_pop = generate_init_pop(piece_edges, N_SEGMENTS, pop_size=6000)

    '''individual = init_pop[0]
    print('First individual in the population:')
    print(individual[0])
    print(individual[0].shape)
    print(individual[1])
    print(individual[2])
    print('Fitness:', get_fitness(individual, N_SEGMENTS))
    visualize(pieces, individual, N_SEGMENTS)'''


    # experimenting with good individuals
    '''fitness_matrix_pairs = np.array([
        get_fitness(ind, N_SEGMENTS) for ind in init_pop])
    fitnesses = np.array([pair[0].sum() + pair[1].sum()
        for pair in fitness_matrix_pairs])

    # analyzing best individual
    best_fitness_index = np.argmin(fitnesses)
    best_ind = init_pop[best_fitness_index]

    print('Best fitness matrix pair:')
    print(fitness_matrix_pairs[best_fitness_index])
    print('Best fitness:', fitnesses[best_fitness_index])

    visualize(pieces, best_ind, N_SEGMENTS)

    # analyzing second best individual
    second_best_fitness_index = np.argmin(
        fitnesses[fitnesses != fitnesses[best_fitness_index]])
    second_best_ind = init_pop[second_best_fitness_index]

    print('Second best fitness matrix pair:')
    print(fitness_matrix_pairs[second_best_fitness_index])
    print('Second best fitness:', fitnesses[second_best_fitness_index])

    visualize(pieces, second_best_ind, N_SEGMENTS)'''
