import numpy as np
import cv2

def get_image_coords_from_vector_index(i, H, W, C):
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

# Example:
# image shape: 64x64 with 3 channels
# index i = 1234
# h, w, c = get_image_coords(1234, 64, 64, 3)
# print(h, w, c)

def get_cost(i, j, H, W, C, scaling_parameter_c=1):
    h1, w1, c1 = get_image_coords_from_vector_index(i, H, W, C)
    h2, w2, c2 = get_image_coords_from_vector_index(j, H, W, C)
    res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
    return res

# Example
# H, W, C = 5, 5, 3
# indices = [12, 13, 14, 15, 16]
# for a in indices:
#     for b in indices:
#         cost = get_cost(a, b, H, W, C)
#         h1, w1, c1 = get_image_coords_from_vector_index(a, H, W, C)
#         h2, w2, c2 = get_image_coords_from_vector_index(b, H, W, C)
#         print(h1, w1, c1)
#         print(h2, w2, c2)
#         print(cost)

def get_cost2(m, i, n, j, H, W, C, scaling_parameter_c=1):
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
