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
# - rotation of the pieces in an ordered 1d-array
def generate_init_pop(piece_edges, n_segments, pop_size=100):
    # indices to keep track of in each individual
    indices = np.arange(len(piece_edges))

    pop = []
    for i in range(pop_size):
        # shuffling the indices and creating an individual with shuffled edges
        shuffled_indices = np.random.permutation(indices)

        individual = []
        rotations = [] # 0: not rotated, 1: 90-degree clock-wise, etc.
        for index in shuffled_indices:
            rotation = np.random.randint(0, 4)
            rotations.append(rotation)
            individual.append(np.roll(piece_edges[index], rotation, axis=0))

        # creating the rotation array
        sorted_rotation_indices = shuffled_indices.argsort()
        rotations = np.array(rotations).flatten()[sorted_rotation_indices]

        # reshaping the indices and individual
        shuffled_indices = shuffled_indices.reshape((n_segments, n_segments))
        individual = np.array(individual).reshape(
            (n_segments, n_segments, 4, -1))
        #rotations = np.array(rotations).reshape((n_segments, n_segments))

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
    piece_indices = ind[1]
    rotations = ind[2]

    print(piece_indices)
    print(rotations)

    f, ax = plt.subplots(n_segments, n_segments, figsize=(5, 5))

    for i in range(n_segments):
        for j in range(n_segments):
            ax[i, j].imshow(
                skimage.transform.rotate(
                    pieces[piece_indices[i, j]],
                    90 * rotations[piece_indices[i, j]]),
                cmap = 'gray')
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


# returns a tuple of the following:
# - cluster matrix: adjacent matching pieces will have the same index
# - cluster fitnesses: average of all differences in adjacent matching pieces in
# each cluster
# - a dictionary mapping a cluster id to a set of piece ids in that cluster
# - match-orientation array: each piece index maps to a 4-element array
# containing either `None` (if the specific side is not matched) or
# (id of match piece, fitness) (if the side is matched)
# 1st element --> match at top, 2nd element --> match on right, etc.
# the orientations are individually-specific and don't correspond to the
# original orientations
def get_ind_stats(ind, threshold, n_segments, fitness_matrix_pair=None):

    # mutates the cluster matrix
    # changes every occurence of the target id to the result id
    def change_cluster_id(cluster_matrix, target_id, result_id):
        for i in range(cluster_matrix.shape[0]):
            for j in range(cluster_matrix.shape[1]):
                if cluster_matrix[i, j] == target_id:
                    cluster_matrix[i, j] = result_id


    # obtaining the fitness matrix
    if fitness_matrix_pair is None:
        fitness_matrix_pair = get_fitness(ind, n_segments)

    # initializing the cluster matrix
    good_match_horizontal_matrix = fitness_matrix_pair[0] <= threshold
    good_match_vertical_matrix = fitness_matrix_pair[1] <= threshold

    cluster_matrix = np.zeros((n_segments, n_segments), dtype=int)

    # initializing the match-orientation array
    piece_indices = ind[1]
    match_orientations = np.array([np.array([None for i in range(4)])
        for j in range(piece_indices.size)])


    # cluster id
    cluster_id_set = set()
    id = 1

    # filling in values in the cluster matrix and match-orientation array
    for i in range(good_match_horizontal_matrix.shape[0]):
        for j in range(good_match_horizontal_matrix.shape[1]):
            if good_match_horizontal_matrix[i, j]:

                if cluster_matrix[i, j] == 0 and cluster_matrix[i, j + 1] == 0:
                    cluster_matrix[i, j] = id
                    cluster_matrix[i, j + 1] = id
                    cluster_id_set.add(id)
                    id += 1

                elif cluster_matrix[i, j] == 0 and cluster_matrix[i, j + 1] != 0:
                    cluster_matrix[i, j] = cluster_matrix[i, j + 1]

                elif cluster_matrix[i, j] != 0 and cluster_matrix[i, j + 1] == 0:
                    cluster_matrix[i, j + 1] = cluster_matrix[i, j]

                else:
                    change_cluster_id(cluster_matrix, cluster_matrix[i, j],
                        cluster_matrix[i, j + 1])

                match_orientations[piece_indices[i, j]][1] = (
                    piece_indices[i, j + 1], fitness_matrix_pair[0][i, j])
                match_orientations[piece_indices[i, j + 1]][3] = (
                    piece_indices[i, j], fitness_matrix_pair[0][i, j])

    for i in range(good_match_vertical_matrix.shape[0]):
        for j in range(good_match_vertical_matrix.shape[1]):
            if good_match_vertical_matrix[i, j]:
                if cluster_matrix[i, j] == 0 and cluster_matrix[i + 1, j] == 0:
                    cluster_matrix[i, j] = id
                    cluster_matrix[i + 1, j] = id
                    cluster_id_set.add(id)
                    id += 1

                elif cluster_matrix[i, j] == 0 and cluster_matrix[i + 1, j] != 0:
                    cluster_matrix[i, j] = cluster_matrix[i + 1, j]

                elif cluster_matrix[i, j] != 0 and cluster_matrix[i + 1, j] == 0:
                    cluster_matrix[i + 1, j] = cluster_matrix[i, j]

                else:
                    change_cluster_id(cluster_matrix, cluster_matrix[i, j],
                        cluster_matrix[i + 1, j])

                match_orientations[piece_indices[i, j]][2] = (
                    piece_indices[i + 1, j], fitness_matrix_pair[1][i, j])
                match_orientations[piece_indices[i + 1, j]][0] = (
                    piece_indices[i, j], fitness_matrix_pair[1][i, j])


    # calculating fitness of each cluster
    # and generating the cluster id - piece set dictionary
    cluster_fitnesses = {}
    cluster_to_piece_set = {}
    for id in cluster_id_set:
        cluster_fitnesses[id] = [0, 0]
        cluster_to_piece_set[id] = set()

    for i in range(cluster_matrix.shape[0]):
        for j in range(cluster_matrix.shape[1]):
            if cluster_matrix[i, j]:
                cluster_to_piece_set[id].add(piece_indices[i, j])

                for item in match_orientations[piece_indices[i, j]]:
                    if item is not None:
                        #print(cluster_matrix[i, j])
                        #print(cluster_fitnesses[cluster_matrix[i, j]])
                        #print(item)
                        cluster_fitnesses[cluster_matrix[i, j]][0] += item[1]
                        cluster_fitnesses[cluster_matrix[i, j]][1] += 1

    for id in cluster_id_set:
        fitness_sum, fitness_count = cluster_fitnesses[id]
        if fitness_count == 0:
            del cluster_fitnesses[id]
        else:
            cluster_fitnesses[id] = fitness_sum / fitness_count


    return (cluster_matrix, cluster_fitnesses, cluster_to_piece_set,
        match_orientations)


# TODO: test
def generate_offspring(parent1, parent2, threshold, n_segments):

    # precondition: intersection is not empty
    # returns True if there is a real conflict
    # returns False if the clusters are mergeable
    def conflict_check(piece_set_intersection):
        for piece_id in piece_set_intersection:
            # subjective orientations (with respect to the current solution/ind)
            parent1_subjective_orientation = parent1_match_orientations[piece_id]
            parent2_subjective_orientation = parent2_match_orientations[piece_id]

            # objective orientations (with respect to the original puzzle)
            parent1_objective_orientation = np.roll(
                parent1_subjective_orientation, -parent1_orientations[piece_id])
            parent2_objective_orientation = np.roll(
                parent2_subjective_orientation, -parent2_orientations[piece_id])

            for i in range(4):
                # return False if the piece is matched with two different pieces
                # in the same direction
                if parent1_objective_orientation[i] is not None and\
                    parent2_objective_orientation[i] is not None and\
                    parent1_objective_orientation[i] != parent2_objective_orientation[i]:

                    print('Conflicting pieces found')
                    print(i)
                    print(parent1_objective_orientation)
                    print(parent2_objective_orientation)
                    return True

        return False

    # generating stats for both parents
    parent1_piece_indices, parent1_orientations = parent1[1], parent1[2]

    parent1_test_fitness_matrix_pair = (np.array([
        [10, 1],
        [10, 10],
        [10, 10]
    ]), np.array([
        [10, 10, 0],
        [10, 10, 10]
    ])) # used for testing

    parent1_cluster_matrix, parent1_cluster_fitnesses, parent1_cluster_to_piece_set,\
        parent1_match_orientations = get_ind_stats(parent1, threshold, n_segments,
            fitness_matrix_pair=parent1_test_fitness_matrix_pair)

    parent2_piece_indices, parent2_orientations = parent2[1], parent2[2]

    parent2_test_fitness_matrix_pair = (np.array([
        [10, 10],
        [10, 10],
        [10, 10]
    ]), np.array([
        [10, 10, 10],
        [1, 10, 10]
    ])) # used for testing

    parent2_cluster_matrix, parent2_cluster_fitnesses, parent2_cluster_to_piece_set,\
        parent2_match_orientations = get_ind_stats(parent2, threshold, n_segments,
            fitness_matrix_pair=parent2_test_fitness_matrix_pair)

    print('Parent1 piece indices:')
    print(parent1_piece_indices)
    print('Parent1 orientations:')
    print(parent1_orientations)
    print('Parent1 fitness matrix pair:')
    print(parent1_test_fitness_matrix_pair[0])
    print(parent1_test_fitness_matrix_pair[1])
    print('Parent1 cluster matrix:')
    print(parent1_cluster_matrix)
    print('Parent1 cluster fitnesses:')
    print(parent1_cluster_fitnesses)
    print('Parent1 cluster to piece:')
    print(parent1_cluster_to_piece_set)
    print('Parent1 match-orientation array:')
    print(parent1_match_orientations)
    print('-' * 50)

    print('Parent2 piece indices:')
    print(parent2_piece_indices)
    print('Parent2 orientations:')
    print(parent2_orientations)
    print('Parent2 fitness matrix pair:')
    print(parent2_test_fitness_matrix_pair[0])
    print(parent2_test_fitness_matrix_pair[1])
    print('Parent2 cluster matrix:')
    print(parent2_cluster_matrix)
    print('Parent2 cluster fitnesses:')
    print(parent2_cluster_fitnesses)
    print('Parent2 cluster to piece:')
    print(parent2_cluster_to_piece_set)
    print('Parent2 match-orientation array:')
    print(parent2_match_orientations)
    print('-' * 50)


    # { ( cluster_id (parent1), cluster_id (parent2) ):
    #    intersection set of piece indices }
    mergeable_clusters = {}
    # list of ( cluster_id (parent1), cluster_id (parent2) )
    conflicted_clusters = []

    for parent1_cluster_id in parent1_cluster_to_piece_set:
        for parent2_cluster_id in parent2_cluster_to_piece_set:
            intersect = parent1_cluster_to_piece_set[parent1_cluster_id].intersection(
                parent2_cluster_to_piece_set[parent2_cluster_id])

            if intersect:
                if conflict_check(intersect):
                    conflicted_clusters.append((parent1_cluster_id,
                        parent2_cluster_id))
                else:
                    mergeable_clusters[(parent1_cluster_id,
                        parent2_cluster_id)] = intersect

    print('Mergeable clusters:')
    print(mergeable_clusters)
    print('Conflicted clusters:')
    print(conflicted_clusters)
    print('-' * 50)
