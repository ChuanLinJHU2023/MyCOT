import matplotlib.pyplot as plt
from distance_causal_transport_version2 import *
from utils import *

down_factor = 32
C=4

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

img = plt.imread('../image_cats/D.jpg')
img_D = np.array(img)
img_D = downsample_image(img_D, down_factor)



# print("################### Causal Distance Between AA #########################")
# dist_AA, _ = calculate_causal_distance_between_images(img_A, img_A, scaling_parameter_c=C)
# print(dist_AA)
# print("#################################################################")


print("################### Causal Distance Between AB #########################")
dist_AB, _ = calculate_causal_distance_between_images_version2(img_A, img_B, scaling_parameter_c=C)
print(dist_AB)
print("#################################################################")


print("################### Causal Distance Between AD #########################")
dist_AD, _ = calculate_causal_distance_between_images_version2(img_A, img_D, scaling_parameter_c=C)
print(dist_AD)
print("#################################################################")