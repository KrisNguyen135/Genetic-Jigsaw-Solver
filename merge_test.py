import numpy as np
np.random.seed(13)

n_segments = 3


# takes in a child's objective match-orientation array and
# returns a random solution that preserves all good clusters
# in the form of (indices, orientations)
def generate_child(child_objective_match_orientations):

    def combine_clusters():

        # global variables for the solution
        indices = np.array([[
            None for _ in range(n_segments)
        ] for __ in range(n_segments)])

        orientations = np.array([
            None for _ in range(n_segments * n_segments)
        ])

        remain_piece_set = set([i for i in range(n_segments * n_segments)])

        # tries to shift all pieces in a cluster in the current
        # solution in a direction
        # direction parameter is clockwise, starting from 0 -> up
        # returns None if shifted
        # returns -1 if conflicting with bounds
        # NA: returns cluster id set if conflicting with other clusters
        def recur_shift_cluster(cluster_id, direction):
            nonlocal indices

            #print(f'Shifting cluster {cluster_id} in direction {direction}')

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
            #print(f'Direction deltas: {row_change}, {col_change}')

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

                        elif indices[new_row, new_col] is not None\
                            and piece_cluster_id[indices[row, col]]\
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
                        print(f'Shifting {indices[row, col]} from ({row}, {col}) to ({new_row}, {new_col})')

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
            if row < 0 or row >= n_segments\
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
                    #print(f'Index of match-orientation: {i}')
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

                '''print(f'{piece_id}, ({row}, {col}) returning:')
                print(overall_row_change + recur_insert_result[0],
                      overall_col_change + recur_insert_result[1])'''

                return (overall_row_change + recur_insert_result[0],
                        overall_col_change + recur_insert_result[1])

        # generating random orientation for each cluster
        cluster_to_orientation = {
            cluster_id: np.random.randint(0, 4) for cluster_id in cluster_id_set
        }

        # creating the subjective match-orientation array
        child_subjective_match_orientations = []
        for i, match_orientation in enumerate(child_objective_match_orientations):
            if piece_cluster_id[i] != 0:
                child_subjective_match_orientations.append(np.roll(
                    match_orientation, cluster_to_orientation[piece_cluster_id[i]]
                ))
            else:
                child_subjective_match_orientations.append(match_orientation)

        child_subjective_match_orientations = np.array(child_subjective_match_orientations)

        print('\nCluster ID to orientation dictionary:')
        print(cluster_to_orientation)
        print('\nChild subjective match-orientation array:')
        print(child_subjective_match_orientations)

        # starting inserting pieces
        print('\nCurrent arrangement:')
        print(indices)
        print('\nRemaining pieces:')
        print(remain_piece_set)
        print(recur_insert_piece(0, 2, 1, 0))

        print('\nCurrent arrangement:')
        print(indices)
        print('\nRemaining pieces:')
        print(remain_piece_set)
        print(recur_insert_piece(2, 1, 0, 0))

        print('\nCurrent arrangement:')
        print(indices)
        print('\nRemaining pieces:')
        print(remain_piece_set)
        print(recur_insert_piece(4, 0, 1, 0))

        print('\nFinal arrangement:')
        print(indices)
        print('\nRemaining pieces:')
        print(remain_piece_set)

        '''for piece_id in range(n_segments * n_segments):
            print('\nCurrent arrangement:')
            print(indices)
            print('\nRemaining pieces:')
            print(remain_piece_set)
            print(f'Attempting to insert Piece {piece_id}')
            print('Result:', recur_insert_piece(piece_id, ))'''


    # assigning matched pieces with the same cluster id
    piece_cluster_id = np.zeros(
        (len(child_objective_match_orientations,)), dtype=int
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
                        piece_cluster_id[piece_id]\
                            = piece_cluster_id[match_piece_id]

                elif piece_cluster_id[match_piece_id] == 0:
                    piece_cluster_id[match_piece_id]\
                        = piece_cluster_id[piece_id]

                elif piece_cluster_id[piece_id]\
                        != piece_cluster_id[match_piece_id]:
                    for i in range(len(piece_cluster_id)):
                        if piece_cluster_id[i]\
                                == piece_cluster_id[match_piece_id]:
                            piece_cluster_id[i]\
                                = piece_cluster_id[piece_id]

    print('\nPiece-wise cluster id array:')
    print(piece_cluster_id)

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

    print('Cluster to piece set:')
    print(cluster_to_piece_set)

    # generating a random solution while preserving good matches
    combine_clusters()


if __name__ == '__main__':
    # non-conflicting case
    child_objective_match_orientations = np.array([
        [None, (1, 0), (3, 1), None],
        [None, None, None, (0, 0)],
        [None, None, (5, 0), None],
        [(0, 1), None, None, None],
        [None, None, (7, 1), None],
        [(2, 0), None, None, None],
        [None, None, None, None],
        [(4, 1), None, None, None],
        [None, None, None, None]
    ])

    generate_child(child_objective_match_orientations)