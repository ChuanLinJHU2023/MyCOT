import ot
import numpy as np

def calculate_OT_distance(vector1, vector2, calculate_cost, reg=None):
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

def calculate_cost_for_image1(i, j, image, distance_between_channel=1, p=2):
    H, W, C = image.shape
    pixel_index_i = i // C
    pixel_index_j = j // C
    color_index_i = i % C
    color_index_j = j % C
    distance_due_to_position = (
        (pixel_index_i // W - pixel_index_j // W) ** p
        + (pixel_index_i % W - pixel_index_j % W) ** p
    )
    distance_due_to_color = (
        distance_between_channel ** p if color_index_i != color_index_j else 0
    )
    distance = distance_due_to_position + distance_due_to_color
    return distance

def calculate_OT_distance_for_image(image1, image2, reg=None, distance_between_channel=1, p=2):
    assert image1.shape == image2.shape
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    calculate_cost = lambda i, j: calculate_cost_for_image1(i, j, image1, distance_between_channel, p)
    res = calculate_OT_distance(vector1, vector2, calculate_cost, reg)
    return res

def calculate_BCOT_distance(Matrix1, Matrix2, calculate_cost, reg=None):
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    outer_cost_matrix = np.zeros((M, N))
    marginal_prob1 = Matrix1.sum(axis=1) / Matrix1.sum()
    marginal_prob2 = Matrix2.sum(axis=1) / Matrix2.sum()
    for m in range(M):
        for n in range(N):
            if marginal_prob1[m] == 0 or marginal_prob2[n] == 0:
                outer_cost_matrix[m, n] = 0
                continue # If marginal prob of m is 0, the cost at (m,n) is meaningless
            inner_cost_matrix = np.zeros((I, J))
            conditional_prob1 = Matrix1[m, :] / Matrix1[m, :].sum()
            conditional_prob2 = Matrix2[n, :] / Matrix2[n, :].sum()
            for i in range(I):
                for j in range(J):
                    inner_cost_matrix[i, j] = calculate_cost(m, n, i, j)
            if reg is None:
                outer_cost_matrix[m, n] = ot.emd2(conditional_prob1, conditional_prob2, inner_cost_matrix)
            else:
                outer_cost_matrix[m, n] = ot.sinkhorn2(conditional_prob1, conditional_prob2, inner_cost_matrix, reg)
    if reg is None:
        cot_dist = ot.emd2(marginal_prob1, marginal_prob2, outer_cost_matrix)
    else:
        cot_dist = ot.sinkhorn2(marginal_prob1, marginal_prob2, outer_cost_matrix, reg)
    return cot_dist

def calculate_cost_for_image2(m, n, i, j, image, distance_between_channel=1, p=2):
    H, W, C = image.shape
    assert m < H * W
    assert n < H * W
    assert i < C
    assert j < C
    distance_due_to_position = (
        (m // W - n // W) ** p + (m % W - n % W) ** p
    )
    distance_due_to_color = (
        distance_between_channel ** p if i != j else 0
    )
    distance = distance_due_to_position + distance_due_to_color
    return distance

def calculate_BCOT_distance_for_image(image1, image2, distance_between_channel=1, p=2, reg=None):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    calculate_cost = lambda m, n, i, j: calculate_cost_for_image2(m, n, i, j, image1, distance_between_channel, p)
    res = calculate_BCOT_distance(Matrix1, Matrix2, calculate_cost, reg)
    return res