import numpy as np

breakthrough_p1_value_array = np.empty((8, 8))
# Why do they go full length? Until 8th row?
breakthrough_p1_value_array = np.array(
    [
        [45.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 45.0],
        [42.0, 43.0, 43.0, 43.0, 43.0, 43.0, 43.0, 42.0],
        [44.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 44.0],
        [47.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 47.0],
        [51.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 51.0],
        [56.0, 61.0, 61.0, 61.0, 61.0, 61.0, 61.0, 56.0],
        [60.0, 68.0, 68.0, 68.0, 68.0, 68.0, 68.0, 60.0],
        [76.0, 76.0, 76.0, 76.0, 76.0, 76.0, 76.0, 76.0],
    ]
)

breakthrough_p2_value_array = np.flip(breakthrough_p1_value_array)


def breakthrough_value_function(state) -> float:
    # Note, the breakthrough string shows the first player as 'b' and the second player as 'w'
    observation_tensor = state.observation_tensor()
    observation_array = np.array(observation_tensor)
    observation_array = observation_array.reshape(4, 8, 8)

    p1_value = (observation_array[0] * breakthrough_p1_value_array).sum()
    p2_value = (observation_array[1] * breakthrough_p2_value_array).sum()
    p1_rel_value = p1_value / (p1_value + p2_value)
    return p1_rel_value


CONNECT_FOUR_COMBO_VALUES = np.array([2, 6, 30])


def connect_four_value_function(state) -> float:
    observation_tensor = state.observation_tensor()
    observation_array = np.array(observation_tensor)
    observation_array = observation_array.reshape(4, 6, 7)
    observation_array = observation_array[:2].astype(int)

    horizontal_segments = sliding_window(observation_array, 4)
    horizontal_scores = evaluate_segments(horizontal_segments)

    vertical_segments = sliding_window(observation_array.transpose((0, 2, 1)), 4)
    vertical_scores = evaluate_segments(vertical_segments)

    diagonal_segments = _get_diagonals_segments(observation_array=observation_array)
    diagonal_scores = evaluate_segments(diagonal_segments)

    scores = horizontal_scores + vertical_scores + diagonal_scores
    net_score = scores[0] - scores[1]

    result = net_score / 1000
    return result


def sliding_window(arr, window_size):
    shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def _get_diagonals_segments(observation_array) -> np.ndarray:
    diagonal_segments_right = np.array(
        [
            np.diagonal(observation_array[:, row : row + 4], diag, axis1=1, axis2=2)
            for row in range(3)
            for diag in range(4)
        ]
    )
    diagonal_segments_right = diagonal_segments_right.transpose(1, 0, 2)

    flipped_observation_array = np.flip(observation_array, axis=2)

    diagonal_segments_left = np.array(
        [
            np.diagonal(flipped_observation_array[:, row : row + 4], diag, axis1=1, axis2=2)
            for row in range(3)
            for diag in range(4)
        ]
    )
    diagonal_segments_left = diagonal_segments_left.transpose(1, 0, 2)
    diagonal_segments = np.concatenate([diagonal_segments_right, diagonal_segments_left], axis=1)
    return diagonal_segments


def evaluate_segments(pieces_in_4) -> np.ndarray:
    p1_pieces_in_4 = np.sum(pieces_in_4[0], axis=-1)
    p2_pieces_in_4 = np.sum(pieces_in_4[1], axis=-1)

    only_p1 = (p1_pieces_in_4 > 0) & (p2_pieces_in_4 == 0)
    only_p2 = (p2_pieces_in_4 > 0) & (p1_pieces_in_4 == 0)

    p1_score = np.sum(CONNECT_FOUR_COMBO_VALUES[p1_pieces_in_4[only_p1] - 1])
    p2_score = np.sum(CONNECT_FOUR_COMBO_VALUES[p2_pieces_in_4[only_p2] - 1])

    return np.array([p1_score, p2_score])


GAME_TO_VALUE_FUNCTION = dict(breakthrough=breakthrough_value_function, connect_four=connect_four_value_function)
