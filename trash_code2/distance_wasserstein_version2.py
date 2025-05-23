import ot
import numpy as np

def calculate_wasser_distance_version2(vector1, vector2, calculate_cost, reg=None):
    I = len(vector1)
    J = len(vector2)
    cost_matrix = np.zeros((I, J))
    marginal_prob1 = vector1 / vector1.sum()
    marginal_prob2 = vector2 / vector2.sum()
    for i in range(I):
        for j in range(J):
            cost_matrix[i, j] = calculate_cost(i, j)
    if reg is None:
        ot_distance = ot.emd2(marginal_prob1, marginal_prob2, cost_matrix)
    else:
        ot_distance = ot.sinkhorn2(marginal_prob1, marginal_prob2, cost_matrix, reg)
    return ot_distance

def get_cost_from_ij_version2(i, j, image, distance_between_channel=4):
    H, W, C = image.shape
    pixel_index_i = i // C
    pixel_index_j = j // C
    color_index_i = i % C
    color_index_j = j % C
    distance_due_to_position = (
        np.abs(pixel_index_i // W - pixel_index_j // W)
        + np.abs(pixel_index_i % W - pixel_index_j % W)
    )
    distance_due_to_color = (
        distance_between_channel if color_index_i != color_index_j else 0
    )
    distance = distance_due_to_position + distance_due_to_color
    return distance

def calculate_wasser_distance_between_images_version2(image1, image2, reg=None, distance_between_channel=4):
    assert image1.shape == image2.shape
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    calculate_cost = lambda i, j: get_cost_from_ij_version2(i, j, image1, distance_between_channel)
    res = calculate_wasser_distance_version2(vector1, vector2, calculate_cost, reg)
    return res, None