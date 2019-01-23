import numpy as np
import skimage

from utils import *

N_SEGMENTS = 3


if __name__ == '__main__':
    # reading in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    pieces = generate_puzzle(img, n_segments=N_SEGMENTS)

    #threshold = generate_threshold(pieces, iqr=True, r=3)
    threshold = generate_threshold(pieces, p=93)
    print('Threshold:', threshold)

    # generating the piece edges, clock-wise
    piece_edges = np.array([np.array([
        piece[0, :],
        piece[:, -1],
        piece[-1, :],
        piece[:, 0]
    ]) for piece in pieces])

    # visualizing the puzzle
    '''visualize_v2(
        pieces,
        (np.arange(N_SEGMENTS * N_SEGMENTS).reshape((N_SEGMENTS, N_SEGMENTS)),
         np.zeros(N_SEGMENTS * N_SEGMENTS)),
        N_SEGMENTS
    )'''

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

    # generating a starting population
    old_pop = generate_init_pop_v2(N_SEGMENTS, pop_size=200)

    sorted_pop = get_sorted_pop(piece_edges, old_pop, threshold, N_SEGMENTS)

    '''print('======Testing first individual======')
    parent1 = sorted_pop[-1]
    print(parent1[0])
    print(parent1[1])
    parent1_fitness_matrix_pair = get_fitness_v2(
        piece_edges, parent1, N_SEGMENTS)
    print(parent1_fitness_matrix_pair[0])
    print(parent1_fitness_matrix_pair[1])
    visualize_v2(pieces, parent1, N_SEGMENTS)

    print('======Testing second individual======')
    parent2 = sorted_pop[-2]
    print(parent2[0])
    print(parent2[1])
    parent2_fitness_matrix_pair = get_fitness_v2(
        piece_edges, parent2, N_SEGMENTS)
    print(parent2_fitness_matrix_pair[0])
    print(parent2_fitness_matrix_pair[1])
    visualize_v2(pieces, parent2, N_SEGMENTS)'''

    # inspecting pairs that don't return
    '''print('Inspecting pairs...')
    parent1 = sorted_pop[191]
    parent2 = sorted_pop[163]
    child = generate_offspring(piece_edges, parent1, parent2, threshold, N_SEGMENTS)
    print(child)'''

    n_iters = 5

    for i in range(n_iters):
        print(f'\n======Generating {i}-th population======')

        new_pop = generate_new_pop(piece_edges, sorted_pop, threshold, N_SEGMENTS,
                                   pop_size=250)

        print('Sorting population...')
        sorted_pop = get_sorted_pop(piece_edges, new_pop, threshold, N_SEGMENTS)

        best_ind = sorted_pop[-1]
        print('Best individual info:')
        print(best_ind[0])
        print(best_ind[1])
        print('Best individual fitness matrix pair:')
        fitness_matrix_pair = get_fitness_v2(piece_edges, best_ind, N_SEGMENTS)
        print(fitness_matrix_pair[0])
        print(fitness_matrix_pair[1])
        visualize_v2(pieces, best_ind, N_SEGMENTS)
