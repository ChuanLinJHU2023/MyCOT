import matplotlib.pyplot as plt
from distance_wasserstein_pulp import *
from utils import *

down_factor = 2
scaling_parameter_c = 4

img = plt.imread('../image_digits/train/0/1-4.png')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_digits/train/0/1-5.png')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

img = plt.imread('../image_digits/train/1/8-4.png')
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