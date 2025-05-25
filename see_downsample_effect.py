import matplotlib.pyplot as plt
from distance_bicausal_pot import *
from utils import *

down_factor = 32
scaling_parameter_c = 4

print("################### First Downsample Method #########################")
def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.array(downsampled_image)

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")




print("################### Second Downsample Method #########################")
def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return np.array(downsampled_image)

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")





print("################### Third Downsample Method #########################")
def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return np.array(downsampled_image)

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")



print("################### Fourth Downsample Method #########################")
def downsample_image(image, factor):
    # Check if the image is a NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    # Downsample by slicing
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/see_downsample_effect.py
# ################### First Downsample Method #########################
# 1.2381343654128578
# ########################################################################
# ################### Second Downsample Method #########################
# 1.2968978966516398
# ########################################################################
# ################### Third Downsample Method #########################
# 1.2990148961046115
# ########################################################################
# ################### Fourth Downsample Method #########################
# 1.2375655077385002
# ########################################################################
#
# Process finished with exit code 0