import numpy as np
import scipy.optimize

def calculate_causal_distance_version3(Matrix1, Matrix2, costs):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming with scipy.optimize.

    Parameters:
    - Matrix1: probabilities for the first distribution (size M x I)
    - Matrix2: probabilities for the second distribution (size N x J)
    - costs: 4D list or array of costs/cost matrix between points (size M x I x N x J)

    Returns:
    - causal_distance: the minimal cost (scalar)
    - transport_plan: the optimal transportation plan matrix (numpy array)
    """
    Matrix1 = np.array(Matrix1)
    Matrix2 = np.array(Matrix2)
    Matrix1 = Matrix1 / np.sum(Matrix1)
    Matrix2 = Matrix2 / np.sum(Matrix2)
    costs = np.array(costs)
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    assert (M, I, N, J) == costs.shape
    assert np.all(Matrix1 >= 0)
    assert np.all(Matrix2 >= 0)

    # Number of variables
    num_vars = M * I * N * J

    # Objective function: minimize total transportation cost
    c = costs.flatten()

    # Equality constraints: marginals must match the distributions and causality
    A_eq = []
    b_eq = []

    # Constraint 1: marginals for Matrix1
    for m in range(M):
        for i in range(I):
            row = np.zeros(num_vars)
            for n in range(N):
                for j in range(J):
                    idx = np.ravel_multi_index((m, i, n, j), (M, I, N, J))
                    row[idx] = 1
            A_eq.append(row)
            b_eq.append(Matrix1[m, i])

    # Constraint 2: marginals for Matrix2
    for n in range(N):
        for j in range(J):
            row = np.zeros(num_vars)
            for m in range(M):
                for i in range(I):
                    idx = np.ravel_multi_index((m, i, n, j), (M, I, N, J))
                    row[idx] = 1
            A_eq.append(row)
            b_eq.append(Matrix2[n, j])

    # Constraint 3: causality
    for m in range(M):
        if np.sum(Matrix1[m]) == 0:
            continue
        for i in range(I):
            for n in range(N):
                row = np.zeros(num_vars)
                conditional_prob_of_i_given_m = Matrix1[m, i] / np.sum(Matrix1[m])
                for j in range(J):
                    idx1 = np.ravel_multi_index((m, i, n, j), (M, I, N, J))
                    row[idx1] = 1
                for ii in range(I):
                    for jj in range(J):
                        idx2 = np.ravel_multi_index((m, ii, n, jj), (M, I, N, J))
                        row[idx2] -= conditional_prob_of_i_given_m
                A_eq.append(row)
                b_eq.append(0)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Bounds: all variables must be non-negative
    bounds = [(0, None) for _ in range(num_vars)]

    # Solve the LP
    result = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Extract the transportation plan
    if result.success:
        transport_plan = result.x.reshape((M, I, N, J))
        causal_distance = result.fun
        return causal_distance, transport_plan
    else:
        print("Optimization failed:", result.message)
        return None, None


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


def calculate_causal_distance_between_images_version3(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([ get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c) for m in range(len(Matrix1)) for i in range(len(Matrix1[0])) for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_causal_distance_version3(Matrix1, Matrix2, costs)
