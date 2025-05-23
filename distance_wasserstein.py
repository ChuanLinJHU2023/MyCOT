import pulp
import numpy as np
from utils import *

def calculate_wasserstein_distance(vector1, vector2, costs):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming.

    Parameters:
    - vector1: list or array of probabilities for the first distribution (size I)
    - vector2: list or array of probabilities for the second distribution (size J)
    - costs: 2D list or array of costs/cost matrix between points (size I x J)

    Returns:
    - wasserstein_distance: the minimal cost (scalar)
    - transport_plan: the optimal transportation plan matrix (numpy array)
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    vector1 = vector1 / np.sum(vector1)
    vector2 = vector2 / np.sum(vector2)
    costs = np.array(costs)
    I = len(vector1)
    J = len(vector2)
    assert (I,J) == costs.shape
    assert np.all(vector1 >= 0)
    assert np.all(vector2 >= 0)
    # Initialize LP problem
    prob = pulp.LpProblem("Wasserstein_Distance", pulp.LpMinimize)
    # Create decision variables for transportation plan T[i,j]
    T = {}
    for i in range(I):
        for j in range(J):
            T[(i, j)] = pulp.LpVariable(f"T_{i}_{j}", lowBound=0)
    # Objective: minimize total transportation cost
    prob += pulp.lpSum([costs[i, j] * T[(i, j)] for i in range(I) for j in range(J)])
    # Constraints: marginals must match the distributions
    for i in range(I):
        prob += pulp.lpSum([T[(i, j)] for j in range(J)]) == vector1[i], f"row_sum_{i}"
    for j in range(J):
        prob += pulp.lpSum([T[(i, j)] for i in range(I)]) == vector2[j], f"col_sum_{j}"
    # Solve the LP
    prob.solve()
    # Retrieve the transportation plan
    transport_plan = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            transport_plan[i, j] = pulp.value(T[(i, j)])
    # Get the optimal value of the problem
    wasserstein_distance = pulp.value(prob.objective)
    return wasserstein_distance, transport_plan


def get_hwc_from_i(i, H, W, C):
    """
    Given the index i in a flattened image vector, return the (h, w, c) position.

    Parameters:
    - i: int, index in the flattened vector
    - H: int, height of the image
    - W: int, width of the image
    - C: int, number of channels

    Returns:
    - (h, w, c): tuple of the pixel's position
    """
    total = H * W * C
    if i < 0 or i >= total:
        raise ValueError("Index i out of bounds for given image dimensions.")

    c = i % C
    i //= C
    w = i % W
    h = i // W
    return h, w, c


def get_cost_from_ij(i, j, H, W, C, scaling_parameter_c=4):
    h1, w1, c1 = get_hwc_from_i(i, H, W, C)
    h2, w2, c2 = get_hwc_from_i(j, H, W, C)
    res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
    return res


def calculate_wasserstein_distance_between_images(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    costs = [[get_cost_from_ij(i, j, H, W, C, scaling_parameter_c) for j in range(len(vector2))] for i in range(len(vector1))]
    return calculate_wasserstein_distance(vector1, vector2, costs)
