import numpy as np
# for testing purposes:
# seed = 0 when N_SEGMENTS = 2 to get perfect solution
# seed = 13 when N_SEGMENTS = 13 to get a solution with a pair of matched pieces
np.random.seed(13)
#import skimage
#import matplotlib.pyplot as plt


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


# returns two 2d arrays (matrices) holding differences between adjacent pieces
# in an individual in a given population
# the first matrix has `n_segments` rows and `n_segments - 1` columns, holding
# differences between horizonally adjacent pieces
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


# drawing the pieces in the order and rotation specified in an individual
def visualize(pieces, ind, n_segments):
    indices = ind[1]
    rotations = ind[2]

    f, ax = plt.subplots(n_segments, n_segments, figsize=(5, 5))

    for i in range(n_segments):
        for j in range(n_segments):
            ax[i, j].imshow(
                skimage.transform.rotate(
                    pieces[indices[i, j]], 90 * rotations[i, j]),
                cmap = 'gray'
            )
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])

    plt.tight_layout()
    plt.show()


# calculates all differences between first and seconds layers of each piece
# returns the percentile threshold of the calculated differences
# used to define potential matches between pairs of pieces
def generate_threshold(pieces, p=100):
    differences = []
    for piece in pieces:
        differences.append(
            np.sum((piece[0, :] - piece[1, :]) ** 2))
        differences.append(
            np.sum((piece[:, -1] - piece[:, -2]) ** 2))
        differences.append(
            np.sum((piece[-1, :] - piece[-2, :]) ** 2))
        differences.append(
            np.sum((piece[:, 0] - piece[:, 1]) ** 2))

    #differences = np.array(differences)

    return np.percentile(differences, p)


def get_ind_stats(ind, threshold, n_segments):

    # returns a cluster matrix and the match-orientation array in a tuple
    #
    # cluster matrix: adjacent matching pieces will have the same index
    #
    # match-orientation array: each piece index maps to a 4-element array
    # containing either `None` (if the specific side is not matched) or
    # `piece_id` (if the side is matched)
    # 1st element --> match at top, 2nd element --> match on right, etc.
    def generate_match_stats(fitness_matrix_pair):

        def change_id(cluster_matrix, target_id, result_id):
            for i in range(cluster_matrix.shape[0]):
                for j in range(cluster_matrix.shape[1]):
                    if cluster_matrix[i, j] == target_id:
                        cluster_matrix[i, j] = result_id


        # initializing the cluster matrix
        good_match_horizontal_matrix = fitness_matrix_pair[0] <= threshold
        good_match_vertical_matrix = fitness_matrix_pair[1] <= threshold

        cluster_matrix = np.zeros((n_segments, n_segments), dtype=int)

        # initializing the match-orientation array
        piece_indices = ind[1]
        rotations = ind[2]
        match_orientations = np.array([np.array([None for i in range(4)])
            for j in range(piece_indices.size)])

        id = 1
        for i in range(good_match_horizontal_matrix.shape[0]):
            for j in range(good_match_horizontal_matrix.shape[1]):
                if good_match_horizontal_matrix[i, j]:

                    if cluster_matrix[i, j] == 0 and cluster_matrix[i, j + 1] == 0:
                        cluster_matrix[i, j] = id
                        cluster_matrix[i, j + 1] = id
                        id += 1
                    elif cluster_matrix[i, j] == 0 and cluster_matrix[i, j + 1] != 0:
                        cluster_matrix[i, j] = cluster_matrix[i, j + 1]
                    elif cluster_matrix[i, j] != 0 and cluster_matrix[i, j + 1] == 0:
                        cluster_matrix[i, j + 1] = cluster_matrix[i, j]
                    else:
                        change_id(cluster_matrix, cluster_matrix[i, j],
                            cluster_matrix[i, j + 1])

                    match_orientations[piece_indices[i, j]][1] = piece_indices[i, j + 1]
                    match_orientations[piece_indices[i, j + 1]][3] = piece_indices[i, j]

        for i in range(good_match_vertical_matrix.shape[0]):
            for j in range(good_match_vertical_matrix.shape[1]):
                if good_match_vertical_matrix[i, j]:
                    if cluster_matrix[i, j] == 0 and cluster_matrix[i + 1, j] == 0:
                        cluster_matrix[i, j] = id
                        cluster_matrix[i + 1, j] = id
                        id += 1
                    elif cluster_matrix[i, j] == 0 and cluster_matrix[i + 1, j] != 0:
                        cluster_matrix[i, j] = cluster_matrix[i + 1, j]
                    elif cluster_matrix[i, j] != 0 and cluster_matrix[i + 1, j] == 0:
                        cluster_matrix[i + 1, j] = cluster_matrix[i, j]
                    else:
                        change_id(cluster_matrix, cluster_matrix[i, j],
                            cluster_matrix[i + 1, j])

                    match_orientations[piece_indices[i, j]][2] = piece_indices[i + 1, j]
                    match_orientations[piece_indices[i + 1, j]][0] = piece_indices[i, j]

        return (cluster_matrix, match_orientations)


    #fitness_matrix_pair = get_fitness(ind, n_segments)
    test_fitness_matrix_pair = (np.array([
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

    cluster_matrix, match_orientations = generate_match_stats(
        test_fitness_matrix_pair)

    print('\nPiece indices:')
    print(ind[1])
    print('\nRotations:')
    print(ind[2])
    print('\nCluster matrix:')
    print(cluster_matrix)
    print('\nMatch-orientation array:')
    print(match_orientations)
    print()

'''def generate_offspring(parent1, fitness_matrix_pair1, parent2, fitness_matrix_pair2,
    threshold, n_segments, n_offsprings=1):

    def generate_cluster_id_to_piece(indices, fitness_matrix_pair, cluster_matrix):
        cluster_id_to_piece = {}

        for i in range(n_segments):
            for j in range(n_segments):
                if cluster_matrix[i, j]:
                    if cluster_matrix[i, j] not in cluster_id_to_piece:
                        cluster_id_to_piece[cluster_matrix[i, j]] = set()

                    cluster_id_to_piece[cluster_matrix[i, j]].add(indices[i, j])

        return cluster_id_to_piece


    # TODO: match clusters that have common pieces
    def find_connected_clusters():

        def get_cluster_fitness(cluster_id, fitness_matrix_pair, cluster_matrix):
            good_match_horizontal_matrix, good_match_vertical_matrix = fitness_matrix_pair

            target_differences = []
            for i in range(good_match_horizontal_matrix.shape[0]):
                for j in range(good_match_horizontal_matrix.shape[1]):
                    if good_match_horizontal_matrix[i, j] <= threshold:
                        if cluster_matrix[i, j] == cluster_id:
                            target_differences.append(good_match_horizontal_matrix[i, j])

            for i in range(good_match_vertical_matrix.shape[0]):
                for j in range(good_match_vertical_matrix.shape[1]):
                    if good_match_vertical_matrix[i, j] <= threshold:
                        if cluster_matrix[i, j] == cluster_id:
                            target_differences.append(good_match_vertical_matrix[i, j])

            return np.array(target_differences).mean()

        cluster_matrix1 = generate_cluster_matrix(fitness_matrix_pair1)
        cluster_matrix2 = generate_cluster_matrix(fitness_matrix_pair2)

        cluster_id_to_piece1 = generate_cluster_id_to_piece(parent1[1],
            fitness_matrix_pair1, cluster_matrix1)
        cluster_id_to_piece2 = generate_cluster_id_to_piece(parent2[1],
            fitness_matrix_pair2, cluster_matrix2)

        #print(get_cluster_fitness(2, fitness_matrix_pair1, cluster_matrix1))

        result_id_to_piece = {}

        for id1 in cluster_id_to_piece1:
            for id2 in cluster_id_to_piece2:
                if cluster_id_to_piece1[id1].intersection(cluster_id_to_piece2[id2]):
                    # TODO: check to see if there is a conflict
                    # if there is, go to the next `if` condition
                    # otherwise merge the two clusters and append to the result dict

                    if get_cluster_fitness(id1, fitness_matrix_pair1, cluster_matrix1)\
                        < get_cluster_fitness(id2, fitness_matrix_pair2, cluster_matrix2):

                        # TODO: append the better cluster to the result dict


    #cluster_matrix = generate_cluster_matrix(fitness_matrix_pair1)
    #print(cluster_matrix)

    #find_connected_clusters()
    return'''
