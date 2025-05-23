import pulp
from utils import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def calculate_causal_distance_version2(Matrix1, Matrix2, costs):
    # Normalize
    Matrix1 = np.array(Matrix1)
    Matrix2 = np.array(Matrix2)
    Matrix1 = Matrix1 / np.sum(Matrix1)
    Matrix2 = Matrix2 / np.sum(Matrix2)
    costs = np.array(costs)
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    assert (M, I, N, J) == costs.shape

    # Initialize Gurobi model
    model = gp.Model("Causal_Distance")
    model.Params.OutputFlag = 0  # silence output for speed

    # Create variables T[m,i,n,j]
    T = {}
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    T[(m,i,n,j)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{m}_{i}_{n}_{j}")

    # Objective: minimize total cost
    model.setObjective(gp.quicksum(costs[m,i,n,j] * T[(m,i,n,j)] for m in range(M) for i in range(I) for n in range(N) for j in range(J)), GRB.MINIMIZE)

    # Marginal constraints
    for m in range(M):
        for i in range(I):
            model.addConstr(
                gp.quicksum(T[(m,i,n,j)] for n in range(N) for j in range(J)) == Matrix1[m,i],
                name=f"marginal_X_m{m}_i{i}"
            )

    for n in range(N):
        for j in range(J):
            model.addConstr(
                gp.quicksum(T[(m,i,n,j)] for m in range(M) for i in range(I)) == Matrix2[n,j],
                name=f"marginal_Y_n{n}_j{j}"
            )

    # Causality constraints
    for m in range(M):
        sum_M1_m = np.sum(Matrix1[m])
        if sum_M1_m == 0:
            continue
        for i in range(I):
            for n in range(N):
                # Sum over j
                sum_j = gp.quicksum(T[(m,i,n,j)] for j in range(J))
                # Sum over i,j pairs
                total_mn = gp.quicksum(T[(m,i,n,j)] for i in range(I) for j in range(J))
                # Conditional probability P_i|m
                p_i_given_m = Matrix1[m,i] / sum_M1_m
                model.addConstr(
                    sum_j == total_mn * p_i_given_m,
                    name=f"causality_m{m}_i{i}_n{n}"
                )

    # Optimize
    model.optimize()

    # Retrieve results
    transport_plan = np.zeros((M, I, N, J))
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    val = T[(m,i,n,j)].X if T[(m,i,n,j)].X is not None else 0
                    transport_plan[m, i, n, j] = val

    causal_distance = model.ObjVal
    return causal_distance, transport_plan


def calculate_causal_distance_between_images_version2(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([ get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c) for m in range(len(Matrix1)) for i in range(len(Matrix1[0])) for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_causal_distance_version2(Matrix1, Matrix2, costs)
