import numpy as np

n_segments = 2

parent1_orientations = np.array([1, 2, 1, 2])
parent1_cluster_to_piece_set = {
    1: {0, 1},
    2: {3, 2}
}

parent2_orientations = np.array([0, 3, 0, 0])
parent2_cluster_to_piece_set = {1: {2, 1}}

piece_orientation = np.array([0, 0, 0, 0])


def cluster_relative_orientation_check():
    for piece_id1 in range(n_segments * n_segments - 1):
        for piece_id2 in range(piece_id1 + 1, n_segments * n_segments):
            child_orientation_d = (piece_orientation[piece_id1] - piece_orientation[piece_id2]) % 4

            if parent1_piece_to_cluster_id[piece_id1] != 0\
                    and parent1_piece_to_cluster_id[piece_id2] != 0\
                    and parent1_piece_to_cluster_id[piece_id1] == parent1_piece_to_cluster_id[piece_id2]:

                parent1_orientation_d = (parent1_orientations[piece_id1] - parent1_orientations[piece_id2]) % 4

                if parent2_piece_to_cluster_id[piece_id1] != 0 \
                        and parent2_piece_to_cluster_id[piece_id2] != 0 \
                        and parent2_piece_to_cluster_id[piece_id1] == parent2_piece_to_cluster_id[piece_id2]:

                    parent2_orientation_d = (parent2_orientations[piece_id1] - parent2_orientations[piece_id2]) % 4

                    if parent1_orientation_d != parent2_orientation_d:
                        return -1

                if child_orientation_d != parent1_orientation_d:
                    print(f'Conflict between {piece_id1} and {piece_id2} with parent1')
                    piece_orientation[piece_id2] = (piece_orientation[piece_id1] - parent1_orientation_d) % 4
                    return False

            if parent2_piece_to_cluster_id[piece_id1] != 0 \
                    and parent2_piece_to_cluster_id[piece_id2] != 0 \
                    and parent2_piece_to_cluster_id[piece_id1] == parent2_piece_to_cluster_id[piece_id2]:

                parent2_orientation_d = (parent2_orientations[piece_id1] - parent2_orientations[piece_id2]) % 4

                if child_orientation_d != parent2_orientation_d:
                    print(f'Conflict between {piece_id1} and {piece_id2} with parent2')
                    piece_orientation[piece_id2] = (piece_orientation[piece_id1] - parent2_orientation_d) % 4
                    return False

    return True


if __name__ == '__main__':
    # preserving relative orientation for good pieces
    parent1_piece_to_cluster_id = {}
    parent2_piece_to_cluster_id = {}
    for cluster_id, piece_set in parent1_cluster_to_piece_set.items():
        for piece_id in piece_set:
            parent1_piece_to_cluster_id[piece_id] = cluster_id
    for cluster_id, piece_set in parent2_cluster_to_piece_set.items():
        for piece_id in piece_set:
            parent2_piece_to_cluster_id[piece_id] = cluster_id
    for piece_id in range(n_segments * n_segments):
        if piece_id not in parent1_piece_to_cluster_id:
            parent1_piece_to_cluster_id[piece_id] = 0
        if piece_id not in parent2_piece_to_cluster_id:
            parent2_piece_to_cluster_id[piece_id] = 0

    print('Parent1 piece to cluster ID:')
    print(parent1_piece_to_cluster_id)
    print('Parent2 piece to cluster ID:')
    print(parent2_piece_to_cluster_id)

    while not cluster_relative_orientation_check():
        print(piece_orientation)
        continue

    print('Final orientation:')
    print(piece_orientation)