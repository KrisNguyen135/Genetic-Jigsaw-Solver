import numpy as np; np.random.seed(13)
import skimage
import matplotlib.pyplot as plt

# returns a 1d array of puzzle pieces
def generate_puzzle(img, n_segments=5, shuffle=True):
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

    if shuffle:
        # shuffling and rotating the pieces
        np.random.shuffle(pieces)
        pieces = np.array([skimage.transform.rotate(
            piece, 90 * np.random.randint(0, 4)) for piece in pieces])

    '''# creating the 2d structure for the pieces
    pieces = pieces.reshape((n_segments, n_segments, segment_size, segment_size))
    #print('Pieces shape:', pieces.shape)'''

    return pieces

# returns a random initial population
# each individual contains a tuple of:
# - piece edge vectors (x number of pieces)
# - indices of the pieces in matrix format
# - rotation of the pieces in matrix format
def generate_init_pop(piece_edges, n_segments, pop_size=100):
    # indices to keep track of in each individual
    indices = np.arange(len(piece_edges))

    pop = []
    for i in range(pop_size):
        # shuffling the indices and creating an individual with shuffled edges
        shuffled_indices = np.random.permutation(indices)

        individual = []
        rotations = []
        for index in shuffled_indices:
            rotation = np.random.randint(0, 4)
            rotations.append(rotation)
            individual.append(np.roll(piece_edges[index], rotation, axis=0))

        # reshaping the indices and individual
        shuffled_indices = shuffled_indices.reshape((n_segments, n_segments))
        individual = np.array(individual).reshape(
            (n_segments, n_segments, 4, -1))
        rotations = np.array(rotations).reshape((n_segments, n_segments))

        pop.append((individual, shuffled_indices, rotations))

    return pop

# returns the differences in piece edges next to each other
def get_fitness(ind, n_segments):
    # simple square of difference
    def get_difference(edge1, edge2):
        return np.sum((edge1 - edge2) ** 2)

    piece_edges = ind[0]
    #running_fitness = 0

    horizontal_fitness_matrix = np.zeros((n_segments, n_segments - 1))
    vertical_fitness_matrix = np.zeros((n_segments - 1, n_segments))

    for i in range(n_segments - 1):
        for j in range(n_segments):
            vertical_fitness_matrix[i, j] = get_difference(
                piece_edges[i, j][2], piece_edges[i + 1, j][0])
            horizontal_fitness_matrix[j, i] = get_difference(
                piece_edges[j, i][1], piece_edges[j, i + 1][3])


    return (horizontal_fitness_matrix, vertical_fitness_matrix)

def visualize(pieces, ind, n_segments):
    indices = ind[1]
    rotations = ind[2]

    f, ax = plt.subplots(n_segments, n_segments, figsize=(5, 5))

    for i in range(n_segments):
        for j in range(n_segments):
            ax[i, j].imshow(
                skimage.transform.rotate(
                    pieces[indices[i, j]], 90 * rotations[i, j]),
                cmap='gray'
            )
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()
