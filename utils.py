import numpy as np
import cv2

def get_image_coords_from_i(i, H, W, C):
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
    h1, w1, c1 = get_image_coords_from_i(i, H, W, C)
    h2, w2, c2 = get_image_coords_from_i(j, H, W, C)
    res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
    return res


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


def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.array(downsampled_image)
