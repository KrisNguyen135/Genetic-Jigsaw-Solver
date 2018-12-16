import numpy as np; np.random.seed(0)
import skimage
import matplotlib.pyplot as plt

# returns a 1d array of puzzle pieces
def generate_puzzle(img, n_segments=5):
    # grayscaling
    img = skimage.color.rgb2gray(img)

    # resizing to a square
    size = min(img.shape) // n_segments * n_segments
    #print('Size:', size)
    img = skimage.transform.resize(img, (size, size), anti_aliasing=True)

    #skimage.io.imshow(img)
    #plt.show()

    # cutting the image into different pieces
    segment_size = size // n_segments
    pieces = np.array([img[
        segment_size * i : segment_size * (i + 1),
        segment_size * j : segment_size * (j + 1)
    ] for i in range(n_segments) for j in range(n_segments)])

    # shuffling and rotating the pieces
    np.random.shuffle(pieces)
    pieces = np.array([skimage.transform.rotate(
        piece, 90 * np.random.randint(0, 4)) for piece in pieces])

    '''# creating the 2d structure for the pieces
    pieces = pieces.reshape((n_segments, n_segments, segment_size, segment_size))
    #print('Pieces shape:', pieces.shape)'''

    # printing out the pieces
    '''f, ax = plt.subplots(n_segments, n_segments, figsize=(10, 10))

    for i in range(n_segments):
        for j in range(n_segments):
            ax[i, j].imshow(pieces[i, j], cmap='gray')
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()'''

    return pieces

# returns a random initial population
# TODO: test
def generate_init_pop(piece_edges, n_segments, pop_size=100):
    # indices to keep track of in each individual
    indices = np.arange(len(piece_edges))

    pop = []
    for i in range(pop_size):
        # shuffling the indices and creating an individual with shuffled edges
        shuffled_indices = np.random.permutation(indices)
        #individual = np.array([skimage.transform.rotate(piece_edges[index],
        #    90 * np.random.randint(0, 4)) for index in shuffled_indices])
        individual = np.array([piece_edges[index] for index in shuffled_indices])
        individual = np.roll(individual, np.random.randint(0, 4))

        # reshaping the indices and individual
        shuffled_indices = shuffled_indices.reshape((n_segments, n_segments))
        individual = individual.reshape((n_segments, n_segments, -1))

        pop.append((individual, shuffled_indices))

    return pop

N_SEGMENTS = 2

if __name__ == '__main__':
    # reading in an image
    img = skimage.io.imread('pictures/michelangelo-creation-of-adam.jpg')

    # generating the puzzle pieces
    pieces = generate_puzzle(img, n_segments=N_SEGMENTS)

    '''print('Individual pieces:')
    for piece in pieces:
        print(piece)
    print(pieces.shape)
    print('*' * 50)'''

    # generating the piece edges, clock-wise
    piece_edges =  np.array([np.array([
        piece[0, :],
        piece[:, -1],
        np.flip(piece[-1, :]),
        np.flip(piece[:, 0])
    ]) for piece in pieces])

    '''print('Piece edges:')
    for piece in piece_edges:
        print(piece)
    print(piece_edges.shape)
    print('*' * 50)'''

    # TODO: test
    # generating the initial random population
    init_pop = generate_init_pop(piece_edges, N_SEGMENTS)
    print(init_pop[0])
