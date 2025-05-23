import pulp
import numpy as np
from utils import *
import ot

def calculate_wasserstein_distance(vector1, vector2, costs):
    marginal_prob1 = vector1 / vector1.sum()
    marginal_prob2 = vector2 / vector2.sum()
    ot_distance = ot.emd2(marginal_prob1, marginal_prob2, costs)
    return ot_distance


def get_cost_from_ij(i, j, H, W, C, scaling_parameter_c=4):
    def get_hwc_from_i(i, H, W, C):
        assert i < H * W * C
        c = i % C
        i //= C
        w = i % W
        h = i // W
        return h, w, c
    h1, w1, c1 = get_hwc_from_i(i, H, W, C)
    h2, w2, c2 = get_hwc_from_i(j, H, W, C)
    res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
    return res


def calculate_wasserstein_distance_between_images(image1, image2, scaling_parameter_c = 4):
    assert image1.shape == image2.shape
    (H, W, C) = image1.shape
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    costs = [[get_cost_from_ij(i, j, H, W, C, scaling_parameter_c) for j in range(len(vector2))] for i in range(len(vector1))]
    res = calculate_wasserstein_distance(vector1, vector2, costs)
    return res, None