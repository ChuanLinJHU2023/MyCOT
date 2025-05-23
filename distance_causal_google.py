from ortools.linear_solver import pywraplp
import numpy as np

def calculate_causal_distance(Matrix1, Matrix2, costs):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming with Google OR-Tools.

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

    # Create the linear solver with the GLOP backend.
    # solver = pywraplp.Solver.CreateSolver('GLOP')
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Create decision variables for transportation plan T[m, i, n, j]
    T = {}
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    T[(m, i, n, j)] = solver.NumVar(0, solver.infinity(), f'T_{m}_{i}_{n}_{j}')

    # Objective: minimize total transportation cost
    objective = solver.Objective()
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    objective.SetCoefficient(T[(m, i, n, j)], float(costs[m, i, n, j]))
    objective.SetMinimization()

    # Constraints: marginals must match the distributions
    for m in range(M):
        for i in range(I):
            constraint = solver.Constraint(float(Matrix1[m, i]), float(Matrix1[m, i]))
            for n in range(N):
                for j in range(J):
                    constraint.SetCoefficient(T[(m, i, n, j)], 1)

    for n in range(N):
        for j in range(J):
            constraint = solver.Constraint(float(Matrix2[n, j]), float(Matrix2[n, j]))
            for m in range(M):
                for i in range(I):
                    constraint.SetCoefficient(T[(m, i, n, j)], 1)

    # Constraints: causality
    for m in range(M):
        if np.sum(Matrix1[m]) == 0:
            continue
        for i in range(I):
            for n in range(N):
                conditional_prob_of_i_given_m = Matrix1[m,i] / np.sum(Matrix1[m])
                constraint = solver.Constraint(0, 0)
                for j in range(J):
                    constraint.SetCoefficient(T[(m, i, n, j)], 1)
                for i2 in range(I):
                    for j in range(J):
                        constraint.SetCoefficient(T[(m, i2, n, j)], -conditional_prob_of_i_given_m)

    # Solve the LP
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # Retrieve the transportation plan
        transport_plan = np.zeros((M, I, N, J))
        for m in range(M):
            for i in range(I):
                for n in range(N):
                    for j in range(J):
                        transport_plan[m, i, n, j] = T[(m, i, n, j)].solution_value()

        # Calculate the causal distance (total cost)
        causal_distance = solver.Objective().Value()

        return causal_distance, transport_plan
    else:
        print('The problem does not have an optimal solution.')
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


def calculate_causal_distance_between_images(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([ get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c) for m in range(len(Matrix1)) for i in range(len(Matrix1[0])) for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_causal_distance(Matrix1, Matrix2, costs)
