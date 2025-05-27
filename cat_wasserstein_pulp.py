import matplotlib.pyplot as plt
from distance_wasserstein_pulp import *
from utils import *

down_factor = 16
scaling_parameter_c = 4

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

img = plt.imread('cat_images/D.jpg')
img_D = np.array(img)
img_D = downsample_image(img_D, down_factor)


print("SHAPE")
print(img_A.shape)


print("################### Wasser Distance Between AB #########################")
dist_AB, _ = calculate_wasserstein_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("#################################################################")


print("################### Wasser Distance Between AD #########################")
dist_AD, _ = calculate_wasserstein_distance_between_images(img_A, img_D, scaling_parameter_c)
print(dist_AD)
print("#################################################################")