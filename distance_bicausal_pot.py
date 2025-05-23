import pulp
from utils import *
import ot

def calculate_bicausal_distance(Matrix1, Matrix2, costs):
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
            inner_cost_matrix = costs[m, :, n, :].reshape((I, J))
            conditional_prob1 = Matrix1[m, :] / Matrix1[m, :].sum()
            conditional_prob2 = Matrix2[n, :] / Matrix2[n, :].sum()
            outer_cost_matrix[m, n] = ot.emd2(conditional_prob1, conditional_prob2, inner_cost_matrix)
    cot_dist = ot.emd2(marginal_prob1, marginal_prob2, outer_cost_matrix)
    return cot_dist


def get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c=4):
    assert m<H*W
    assert n<H*W
    assert i<C
    assert j<C
    h1 = m // W
    w1 = m % W
    h2 = n // W
    w2 = n % W
    c1 = i
    c2 = j
    res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
    return res


def calculate_bicausal_distance_between_images(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([ get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c) for m in range(len(Matrix1)) for i in range(len(Matrix1[0])) for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_bicausal_distance(Matrix1, Matrix2, costs), None
