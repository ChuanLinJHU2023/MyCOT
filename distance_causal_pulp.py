import pulp
from utils import *


def calculate_causal_distance(Matrix1, Matrix2, costs):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming.
    Note that we transport from Matrix 1 to Matrix 2
    In other words, Matrix 1 is P(X^, Y^) while Matrix 2 is P(X, Y)

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
    prob = pulp.LpProblem("Causal_Distance", pulp.LpMinimize)
    T = {}
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    T[(m, i, n, j)] = pulp.LpVariable(f"T_{m}_{i}_{n}_{j}", lowBound=0)
    # Objective: minimize total transportation cost
    prob += pulp.lpSum([costs[m, i, n, j] * T[(m, i, n, j)] for m in range(M) for i in range(I) for n in range(N) for j in range(J)])
    # Constraints: marginals must match the distributions
    for m in range(M):
        for i in range(I):
            # P~(X^ = m , Y^ = i) == P(X^ = m, Y^ = i)
            prob += pulp.lpSum([T[(m, i, n, j)] for n in range(N) for j in range(J)]) == Matrix1[m, i], f"marginal_prob_Xhat_Yhat__{m}_{i}"
    for n in range(N):
        for j in range(J):
            # P~(X = n , Y = j) == P(X = n, Y = j)
            prob += pulp.lpSum([T[(m, i, n, j)] for m in range(M) for i in range(I)]) == Matrix2[n, j], f"marginal_prob_X_Y_{n}_{j}"
    # Constraints: causality
    # Given X^, X is independent of Y
    for m in range(M):
        if np.sum(Matrix1[m]) == 0:
            # in the case, P(X^ = m) = 0 and P(Y^ = i | X^ = m) is undefined. Causality is satisfied automatically
            continue
        for i in range(I):
            for n in range(N):
                conditional_prob_of_i_given_m = Matrix1[m,i] / np.sum(Matrix1[m])
                # P~(X^ = m , Y^ = i, X = n) == P~(X^ = m, X = n) * P(Y^ = i | X^ = m)
                prob += \
                    pulp.lpSum([T[(m, i, n, j)] for j in range(J)]) \
                    == \
                    pulp.lpSum([T[(m, i, n, j)] for i in range(I) for j in range(J)]) * conditional_prob_of_i_given_m \
                    , f"causality_{m}_{i}_{n}"
    prob.solve(pulp.GUROBI())
    transport_plan = np.zeros((M, I, N, J))
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    transport_plan[m, i, n, j] = pulp.value(T[(m, i, n, j)])
    causal_distance = pulp.value(prob.objective)
    return causal_distance, transport_plan


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


def calculate_causal_distance_between_images(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([ get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c) for m in range(len(Matrix1)) for i in range(len(Matrix1[0])) for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_causal_distance(Matrix1, Matrix2, costs)

