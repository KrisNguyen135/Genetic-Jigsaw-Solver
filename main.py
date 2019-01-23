from utils import *

N_SEGMENTS = 3


if __name__ == '__main__':
    # reading in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    pieces = generate_puzzle(img, n_segments=N_SEGMENTS)

    # getting threshold to specify a good match
    # can be tuned with parameter `p`
    threshold = generate_threshold(pieces, p=93)
    print('Threshold:', threshold)

    # generating the piece edges, clock-wise
    piece_edges = np.array([np.array([
        piece[0, :],
        piece[:, -1],
        piece[-1, :],
        piece[:, 0]
    ]) for piece in pieces])

    # generating a starting population and sorting it
    old_pop = generate_init_pop_v2(N_SEGMENTS, pop_size=200)
    sorted_pop = get_sorted_pop(piece_edges, old_pop, threshold, N_SEGMENTS)

    # beginning evolution
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
        visualize_v2(pieces, best_ind, N_SEGMENTS, output=f'output/{i}-iter.png')
