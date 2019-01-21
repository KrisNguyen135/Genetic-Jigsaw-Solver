import numpy as np
# for testing purposes:
# seed = 0 when N_SEGMENTS = 2 to get perfect solution
# seed = 13 when N_SEGMENTS = 3 to get a solution with a pair of matched pieces
np.random.seed(0)
#import skimage
#import matplotlib.pyplot as plt
import random


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
    # indices to keep track of each individual
    indices = np.arange(len(piece_edges))

    pop = []
    for i in range(pop_size):
        # shuffling the indices and creating an individual with shuffled edges
        shuffled_indices = np.random.permutation(indices)

        individual = []
        rotations = []  # 0: not rotated, 1: 90-degree clock-wise, etc.
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

        pop.append((individual, shuffled_indices, rotations))

    return pop


# TODO: test to see if the same results as v1 can be produced (in main)
# returns a random initial population
# each individual contains a tuple of:
# - indices of the pieces in matrix format
# - orientations of the pieces in an ordered 1d-array
def generate_init_pop_v2(n_segments, pop_size=100):
    # indices to keep track of each individual
    indices = np.arange(n_segments * n_segments)

    pop = []
    for i in range(pop_size):
        # shuffling the indices
        shuffled_indices = np.random.permutation(indices)

        orientations = []  # 0: not rotated, 1: 90-degree clock-wise, etc.
        for index in shuffled_indices:
            orientation = np.random.randint(0, 4)
            orientations.append(orientation)

        # creating the orientation array
        sorted_orientation_indices = shuffled_indices.argsort()
        orientations = np.array(
            orientations
        ).flatten()[sorted_orientation_indices]

        # reshaping the indices
        shuffled_indices = shuffled_indices.reshape((n_segments, n_segments))

        pop.append((shuffled_indices, orientations))

    return pop


# returns two 2d arrays (matrices) holding differences between adjacent pieces
# in an individual in a given population
# the first matrix has `n_segments` rows and `n_segments - 1` columns, holding
# differences between horizontally adjacent pieces
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

    return horizontal_fitness_matrix, vertical_fitness_matrix


# TODO: test to see if the same results as v1 can be produced (in main)
# returns two 2d-arrays (matrices) holding differences between adjacent pieces
# in an individual in a given population
# first matrix has `n_segments` rows and `n_segments - 1` columns, holding
# differences between horizontally adjacent pieces
def get_fitness_v2(piece_edges, ind, n_segments):
    # simple squares of differences
    def get_difference(edge1, edge2):
        return np.sum((edge1 - edge2) ** 2)

    ind_piece_edges = []
    for piece_id, orientation in zip(ind[0], ind[1]):
        ind_piece_edges.append(
            np.roll(piece_edges[piece_id], orientation, axis=0))

    horizontal_fitness_matrix = np.zeros((n_segments, n_segments - 1))
    vertical_fitness_matrix = np.zeros((n_segments - 1, n_segments))

    for i in range(n_segments - 1):
        for j in range(n_segments):
            vertical_fitness_matrix[i, j] = get_difference(
                ind_piece_edges[i, j][2], ind_piece_edges[i + 1, j][0])
            horizontal_fitness_matrix[j, i] = get_difference(
                ind_piece_edges[j, i][1], ind_piece_edges[j, i + 1][3])

    return horizontal_fitness_matrix, vertical_fitness_matrix


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
                cmap='gray')
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


# mutates the cluster matrix
# changes every occurence of the target id to the result id
def change_cluster_id(cluster_matrix, target_id, result_id):
    for i in range(cluster_matrix.shape[0]):
        for j in range(cluster_matrix.shape[1]):
            if cluster_matrix[i, j] == target_id:
                cluster_matrix[i, j] = result_id


# returns a tuple of the following:
# - cluster matrix: adjacent matching pieces will have the same index
# - cluster id set: set of cluster ids
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
    # obtaining the fitness matrix
    if fitness_matrix_pair is None:
        fitness_matrix_pair = get_fitness(ind, n_segments)

    # initializing the cluster matrix
    good_match_horizontal_matrix = fitness_matrix_pair[0] <= threshold
    good_match_vertical_matrix = fitness_matrix_pair[1] <= threshold

    cluster_matrix = np.zeros((n_segments, n_segments), dtype=int)

    # initializing the match-orientation array
    piece_indices = ind[1]
    match_orientations = np.array([
        np.array([None for _ in range(4)]) for __ in range(piece_indices.size)
    ])

    # initializing cluster id
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

                elif cluster_matrix[i, j] != cluster_matrix[i, j + 1]:
                    cluster_id_set.remove(cluster_matrix[i, j])
                    change_cluster_id(
                        cluster_matrix,
                        cluster_matrix[i, j], cluster_matrix[i, j + 1]
                    )

                match_orientations[piece_indices[i, j]][1] = (
                    piece_indices[i, j + 1], fitness_matrix_pair[0][i, j]
                )
                match_orientations[piece_indices[i, j + 1]][3] = (
                    piece_indices[i, j], fitness_matrix_pair[0][i, j]
                )

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

                elif cluster_matrix[i, j] != cluster_matrix[i + 1, j]:
                    cluster_id_set.remove(cluster_matrix[i, j])
                    change_cluster_id(
                        cluster_matrix,
                        cluster_matrix[i, j], cluster_matrix[i + 1, j]
                    )

                match_orientations[piece_indices[i, j]][2] = (
                    piece_indices[i + 1, j], fitness_matrix_pair[1][i, j]
                )
                match_orientations[piece_indices[i + 1, j]][0] = (
                    piece_indices[i, j], fitness_matrix_pair[1][i, j]
                )

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
                        cluster_fitnesses[cluster_matrix[i, j]][0] += item[1]
                        cluster_fitnesses[cluster_matrix[i, j]][1] += 1

    for id in cluster_id_set:
        fitness_sum, fitness_count = cluster_fitnesses[id]
        if fitness_count == 0:
            del cluster_fitnesses[id]
        else:
            cluster_fitnesses[id] = fitness_sum / fitness_count

    return (cluster_matrix, cluster_id_set, cluster_fitnesses,
            cluster_to_piece_set, match_orientations)


# returns a randomly generated child that preserves all good
# matches from each parent and attempts to merge any mergeable
# clusters
def generate_offspring(parent1, parent2, threshold, n_segments):

    def print_parents_info():
        print('Parent1 piece indices:')
        print(parent1_piece_indices)
        print('Parent1 orientations:')
        print(parent1_orientations)
        print('Parent1 fitness matrix pair:')
        print(parent1_test_fitness_matrix_pair[0])
        print(parent1_test_fitness_matrix_pair[1])
        print('Parent1 cluster matrix:')
        print(parent1_cluster_matrix)
        print('Parent1 cluster id set:')
        print(parent1_cluster_id_set)
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
        print('Parent2 cluster id set:')
        print(parent2_cluster_id_set)
        print('Parent2 cluster fitnesses:')
        print(parent2_cluster_fitnesses)
        print('Parent2 cluster to piece:')
        print(parent2_cluster_to_piece_set)
        print('Parent2 match-orientation array:')
        print(parent2_match_orientations)
        print('-' * 50)

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

                    return True

        return False

    # remove information regarding a specific cluster id from all ind stats
    def remove_cluster(cluster_id, piece_indices, cluster_matrix, cluster_id_set,
                       cluster_fitnesses, cluster_to_piece_set, match_orientations):

        for i in range(cluster_matrix.shape[0]):
            for j in range(cluster_matrix.shape[1]):
                if cluster_matrix[i, j] == cluster_id:
                    match_orientations[piece_indices[i, j]] = np.array(
                        [None for i in range(4)]
                    )

        change_cluster_id(cluster_matrix, cluster_id, 0)
        cluster_id_set.remove(cluster_id)
        del cluster_fitnesses[cluster_id]
        del cluster_to_piece_set[cluster_id]

    # takes in a child's objective match-orientation array and
    # returns a random solution that preserves all good clusters
    # in the form of (indices, orientations)
    def generate_child(child_objective_match_orientations):

        def combine_clusters():
            # nonlocal variables for the solution
            indices = np.array([[
                None for _ in range(n_segments)
            ] for __ in range(n_segments)])

            remain_piece_set = set([i for i in range(n_segments * n_segments)])

            # tries to shift all pieces in a cluster in the current
            # solution in a direction
            # direction parameter is clockwise, starting from 0 -> up
            # returns None if shifted
            # returns -1 if conflicting with bounds
            # NA: returns cluster id set if conflicting with other clusters
            def recur_shift_cluster(cluster_id, direction):
                nonlocal indices

                # print(f'Shifting cluster {cluster_id} in direction {direction}')

                # finding appropriate delta x and y
                row_change = 0
                col_change = 0
                if direction == 0:
                    row_change -= 1
                elif direction == 1:
                    col_change += 1
                elif direction == 2:
                    row_change += 1
                else:
                    col_change -= 1
                # print(f'Direction deltas: {row_change}, {col_change}')

                # finding potential conflicts
                bound_conflict = False
                cluster_conflict = False
                conflicting_cluster_id_set = set()

                for row in range(n_segments):
                    if bound_conflict:
                        break

                    for col in range(n_segments):
                        if indices[row, col] in cluster_to_piece_set[cluster_id]:
                            new_row = row + row_change
                            new_col = col + col_change

                            if new_row < 0 or new_row >= n_segments:
                                bound_conflict = True
                                break
                            elif new_col < 0 or new_col >= n_segments:
                                bound_conflict = True
                                break

                            elif indices[new_row, new_col] is not None \
                                    and piece_cluster_id[indices[row, col]] \
                                    != piece_cluster_id[indices[new_row, new_col]]:

                                cluster_conflict = True
                                conflicting_cluster_id_set.add(
                                    piece_cluster_id[indices[new_row, new_col]]
                                )

                # handling potential conflicts
                if bound_conflict:
                    return -1
                if cluster_conflict:
                    for conflicting_cluster_id in conflicting_cluster_id_set:
                        recur_shift_result = recur_shift_cluster(
                            conflicting_cluster_id, direction
                        )
                        if recur_shift_result == -1:
                            return -1

                # if there is no conflict
                # shifting all targeted pieces
                new_indices = np.copy(indices)
                for row in range(n_segments):
                    for col in range(n_segments):
                        if indices[row, col] in cluster_to_piece_set[cluster_id]:
                            new_row = row + row_change
                            new_col = col + col_change

                            #print(f'Shifting {indices[row, col]} from ({row}, {col}) to ({new_row}, {new_col})')

                            new_indices[new_row, new_col] = indices[row, col]
                            if new_indices[row, col] == indices[row, col]:
                                new_indices[row, col] = None

                indices = new_indices

            # returns None if successful
            # returns (row_change, col_change) if successful but shifted
            # returns -1 if the insertion is recursively impossible
            def recur_insert_piece(piece_id, row, col, direction):
                nonlocal indices

                if piece_id not in remain_piece_set:
                    return

                # if out of bound, shift the whole cluster
                # in opposite direction
                if row < 0 or row >= n_segments \
                        or col < 0 or col >= n_segments:

                    recur_shift_result = recur_shift_cluster(
                        piece_cluster_id[piece_id],
                        (direction + 2) % 4
                    )

                    if recur_shift_result == -1:
                        return -1

                    overall_row_change = 0
                    overall_col_change = 0
                    # adjusting the current location
                    if direction == 0:
                        overall_row_change += 1
                    elif direction == 1:
                        overall_col_change -= 1
                    elif direction == 2:
                        overall_row_change -= 1
                    else:
                        overall_col_change += 1

                    # attempts to insert the piece again
                    # after the shift
                    if direction == 0:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row + 1, col, 0
                        )
                    elif direction == 1:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row, col - 1, 1
                        )
                    elif direction == 2:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row - 1, col, 2
                        )
                    else:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row, col + 1, 3
                        )

                    if recur_insert_result == -1:
                        return -1

                    '''print(f'{piece_id}, ({row}, {col}) returning:')
                    print(overall_row_change + recur_insert_result[0],
                          overall_col_change + recur_insert_result[1])'''

                    return (overall_row_change + recur_insert_result[0],
                            overall_col_change + recur_insert_result[1])

                # if the current cell is empty
                if indices[row, col] is None:
                    indices[row, col] = piece_id
                    remain_piece_set.remove(piece_id)

                    overall_row_change = 0
                    overall_col_change = 0
                    for i in range(4):
                        # print(f'Index of match-orientation: {i}')
                        match_orientation = child_subjective_match_orientations[piece_id][i]

                        if match_orientation is not None:
                            match_piece_id, _ = match_orientation
                            if i == 0:
                                recur_insert_result = recur_insert_piece(
                                    match_piece_id, row - 1, col, 0
                                )
                            elif i == 1:
                                recur_insert_result = recur_insert_piece(
                                    match_piece_id, row, col + 1, 1
                                )
                            elif i == 2:
                                recur_insert_result = recur_insert_piece(
                                    match_piece_id, row + 1, col, 2
                                )
                            else:
                                recur_insert_result = recur_insert_piece(
                                    match_piece_id, row, col - 1, 3
                                )

                            if recur_insert_result == -1:
                                return -1

                            # adjusting the current location
                            if recur_insert_result is not None:
                                row_change, col_change = recur_insert_result

                                overall_row_change += row_change
                                overall_col_change += col_change

                                row += row_change
                                col += col_change

                    '''print(f'{piece_id}, ({row}, {col}) returning:')
                    print(overall_row_change, overall_col_change)'''

                    return overall_row_change, overall_col_change

                # if there is a conflict with another cluster
                else:
                    saved_indices = np.copy(indices)

                    # try shifting the conflicting cluster
                    # in the same direction
                    recur_shift_result = recur_shift_cluster(
                        piece_cluster_id[indices[row, col]], direction
                    )
                    # if successful, attempts to insert the piece again
                    # after the shift
                    if recur_shift_result is None:
                        return recur_insert_piece(piece_id, row, col, direction)

                    # if failed, try shifting the original cluster
                    # in the opposite direction
                    indices = saved_indices
                    recur_shift_result = recur_shift_cluster(
                        piece_cluster_id[piece_id],
                        (direction + 2) % 4
                    )

                    if recur_shift_result == -1:
                        return -1

                    overall_row_change = 0
                    overall_col_change = 0
                    # adjusting the current location
                    if direction == 0:
                        overall_row_change += 1
                    elif direction == 1:
                        overall_col_change -= 1
                    elif direction == 2:
                        overall_row_change -= 1
                    else:
                        overall_col_change += 1

                    # attempts to insert the piece again
                    # after the shift
                    if direction == 0:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row + 1, col, 0
                        )
                    elif direction == 1:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row, col - 1, 1
                        )
                    elif direction == 2:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row - 1, col, 2
                        )
                    else:
                        recur_insert_result = recur_insert_piece(
                            piece_id, row, col + 1, 3
                        )

                    if recur_insert_result == -1:
                        return -1

                    '''print(f'{piece_id}, ({row}, {col}) returning:')
                    print(overall_row_change + recur_insert_result[0],
                          overall_col_change + recur_insert_result[1])'''

                    return (overall_row_change + recur_insert_result[0],
                            overall_col_change + recur_insert_result[1])

            # generating random orientation for each cluster
            cluster_to_orientation = {
                cluster_id: np.random.randint(0, 4) for cluster_id in cluster_id_set
            }
            piece_orientation = np.array([
                cluster_to_orientation[piece_cluster_id[piece_id]] if piece_cluster_id[piece_id] != 0
                else np.random.randint(0, 4) for piece_id in range(n_segments * n_segments)
            ])

            # creating the subjective match-orientation array
            child_subjective_match_orientations = []
            for i, match_orientation in enumerate(child_objective_match_orientations):
                child_subjective_match_orientations.append(np.roll(
                    match_orientation, piece_orientation[i]
                ))

            child_subjective_match_orientations = np.array(child_subjective_match_orientations)

            '''print('\nCluster ID to orientation dictionary:')
            print(cluster_to_orientation)
            print('\nPiece to orientation:')
            print(piece_orientation)
            print('\nChild subjective match-orientation array:')
            print(child_subjective_match_orientations)'''

            # start inserting pieces in order of cluster
            for piece_id in range(n_segments * n_segments):
                if piece_cluster_id[piece_id] != 0:
                    remain_locations = set(
                        map(tuple, np.argwhere(indices == None))
                    )

                    recur_insert_result = -1
                    saved_indices = indices
                    saved_remain_piece_set = remain_piece_set

                    while remain_locations and recur_insert_result == -1:
                        '''print('\nCurrent arrangement:')
                        print(indices)
                        print('\nRemaining pieces:')
                        print(remain_piece_set)
                        print('\nRemaining locations:')
                        print(remain_locations)'''

                        location = random.sample(remain_locations, 1)[0]
                        remain_locations.remove(location)
                        row, col = location

                        indices = np.copy(saved_indices)
                        remain_piece_set = saved_remain_piece_set.copy()

                        # print(f'\nAttempting to insert Piece {piece_id} in ({row}, {col})')
                        recur_insert_result = recur_insert_piece(piece_id, row, col, 0)
                        # print('Result:', recur_insert_result)

                    if recur_insert_result == -1:
                        return -1

            # inserting the rest of the pieces
            # (pieces w/o any matches)
            for piece_id in range(n_segments * n_segments):
                if piece_cluster_id[piece_id] == 0:
                    remain_locations = set(
                        map(tuple, np.argwhere(indices == None))
                    )

                    location = random.sample(remain_locations, 1)[0]
                    remain_locations.remove(location)
                    row, col = location

                    recur_insert_piece(piece_id, row, col, 0)

            '''print('\nFinal arrangement:')
            print(indices)
            print('\nRemaining pieces:')
            print(remain_piece_set)'''

            return indices, piece_orientation

        # assigning matched pieces with the same cluster id
        piece_cluster_id = np.zeros(
            (len(child_objective_match_orientations, )), dtype=int
        )

        id = 1
        for piece_id in range(len(child_objective_match_orientations)):
            for item in child_objective_match_orientations[piece_id]:
                if item is not None:
                    match_piece_id, _ = item
                    if piece_cluster_id[piece_id] == 0:
                        if piece_cluster_id[match_piece_id] == 0:
                            piece_cluster_id[piece_id] = id
                            piece_cluster_id[match_piece_id] = id
                            id += 1
                        else:
                            piece_cluster_id[piece_id] \
                                = piece_cluster_id[match_piece_id]

                    elif piece_cluster_id[match_piece_id] == 0:
                        piece_cluster_id[match_piece_id] \
                            = piece_cluster_id[piece_id]

                    elif piece_cluster_id[piece_id] \
                            != piece_cluster_id[match_piece_id]:
                        for i in range(len(piece_cluster_id)):
                            if piece_cluster_id[i] \
                                    == piece_cluster_id[match_piece_id]:
                                piece_cluster_id[i] \
                                    = piece_cluster_id[piece_id]

        #print('\nPiece-wise cluster id array:')
        #print(piece_cluster_id)

        # generating necessary data structures
        cluster_id_set = set(piece_cluster_id)
        cluster_id_set.remove(0)

        cluster_to_piece_set = {}
        for piece_id in range(n_segments * n_segments):
            temp_cluster_id = piece_cluster_id[piece_id]
            if temp_cluster_id != 0:
                if temp_cluster_id not in cluster_to_piece_set:
                    cluster_to_piece_set[temp_cluster_id] = {piece_id}
                else:
                    cluster_to_piece_set[temp_cluster_id].add(piece_id)

        #print('Cluster to piece set:')
        #print(cluster_to_piece_set)

        # generating a random solution while preserving good matches
        return combine_clusters()

    # generating stats for both parents
    parent1_piece_indices, parent1_orientations = parent1[1], parent1[2]

    parent1_test_fitness_matrix_pair = (np.array([
        [10, 1],
        [10, 10],
        [10, 10]
    ]), np.array([
        [10, 10, 0],
        [10, 10, 10]
    ]))  # used for testing

    parent1_cluster_matrix, parent1_cluster_id_set, parent1_cluster_fitnesses,\
        parent1_cluster_to_piece_set, parent1_match_orientations\
        = get_ind_stats(
            parent1, threshold, n_segments,
            fitness_matrix_pair = parent1_test_fitness_matrix_pair
        )

    parent2_piece_indices, parent2_orientations = parent2[1], parent2[2]

    parent2_test_fitness_matrix_pair = (np.array([
        [10, 10],
        [10, 0], # non-conflicting case
        #[10, 10], # mergeable and conflicting cases
        [10, 10]
    ]), np.array([
        [10, 10, 10],
        [1, 10, 10] # non-conflicting and mergeable cases
        #[1.6, 10, 10] # conflicting case
    ]))  # used for testing

    # large cluster case 1
    '''parent2_test_fitness_matrix_pair = (np.array([
        [10, 1],
        [10, 0],
        [10, 10]
    ]), np.array([
        [10, 0.5, 0],
        [10, 10, 10]
    ]))  # used for testing'''

    # large cluster case 2
    '''parent2_test_fitness_matrix_pair = (np.array([
        [10, 10],
        [1, 10],
        [0, 0]
    ]), np.array([
        [10, 10, 10],
        [0.5, 0, 10]
    ]))  # used for testing'''

    parent2_cluster_matrix, parent2_cluster_id_set, parent2_cluster_fitnesses,\
        parent2_cluster_to_piece_set, parent2_match_orientations\
        = get_ind_stats(
            parent2, threshold, n_segments,
            fitness_matrix_pair = parent2_test_fitness_matrix_pair
        )

    print_parents_info()

    conflicted_clusters = []

    # for every pair of parent1 cluster id and parent2 cluster id,
    # check to see if there is a non-empty intersection between the two piece sets,
    # if so, check to see if there is a conflict in the two sets
    for parent1_cluster_id in parent1_cluster_to_piece_set:
        for parent2_cluster_id in parent2_cluster_to_piece_set:
            intersect = parent1_cluster_to_piece_set[parent1_cluster_id].intersection(
                parent2_cluster_to_piece_set[parent2_cluster_id])

            if intersect and conflict_check(intersect):
                conflicted_clusters.append(
                    (parent1_cluster_id, parent2_cluster_id)
                )

    #print('Mergeable clusters:')
    #print(mergeable_clusters)
    print('Conflicted clusters:')
    print(conflicted_clusters)
    print('-' * 50)

    # keeping track of bad clusters and remove them from ind stats of both parents
    parent1_clusters_to_remove = set()
    parent2_clusters_to_remove = set()
    for parent1_cluster_id, parent2_cluster_id in conflicted_clusters:
        if parent1_cluster_fitnesses[parent1_cluster_id]\
                < parent2_cluster_fitnesses[parent2_cluster_id]:

            parent2_clusters_to_remove.add(parent2_cluster_id)
        else:
            parent1_clusters_to_remove.add(parent1_cluster_id)

    print('Parent1 clusters to remove:')
    print(parent1_clusters_to_remove)
    print('Parent2 clusters to remove:')
    print(parent2_clusters_to_remove)

    for cluster_id in parent1_clusters_to_remove:
        remove_cluster(
            cluster_id, parent1_piece_indices, parent1_cluster_matrix,
            parent1_cluster_id_set, parent1_cluster_fitnesses,
            parent1_cluster_to_piece_set, parent1_match_orientations
        )
    for cluster_id in parent2_clusters_to_remove:
        remove_cluster(
            cluster_id, parent2_piece_indices, parent2_cluster_matrix,
            parent2_cluster_id_set, parent2_cluster_fitnesses,
            parent2_cluster_to_piece_set, parent2_match_orientations
        )

    print('======After removing bad clusters======')
    print_parents_info()

    # transferring match-orientation arrays from parents to child
    child_objective_match_orientations = np.array(
        [np.array([None for i in range(4)])
            for j in range(n_segments * n_segments)]
    )

    for piece_id in range(n_segments * n_segments):
        # subjective orientations (with respect to the current solution/ind)
        parent1_subjective_orientation = parent1_match_orientations[piece_id]
        parent2_subjective_orientation = parent2_match_orientations[piece_id]

        # objective orientations (with respect to the original puzzle)
        parent1_objective_orientation = np.roll(
            parent1_subjective_orientation, -parent1_orientations[piece_id]
        )
        parent2_objective_orientation = np.roll(
            parent2_subjective_orientation, -parent2_orientations[piece_id]
        )

        for i in range(4):
            if parent1_objective_orientation[i] is not None:
                child_objective_match_orientations[piece_id][i]\
                    = parent1_objective_orientation[i]

            if parent2_objective_orientation[i] is not None:
                child_objective_match_orientations[piece_id][i]\
                    = parent2_objective_orientation[i]

    print('======Child objective match-orientation array======')
    print(child_objective_match_orientations)

    child_result = -1

    while child_result == -1:
        child_result = generate_child(child_objective_match_orientations)

    '''indices, orientations = child_result

    print('Generated child indices:')
    print(indices)
    print('Generated child orientations:')
    print(orientations)

    return indices, orientations'''

    return child_result
