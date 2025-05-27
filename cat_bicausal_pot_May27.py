import matplotlib.pyplot as plt
from distance_bicausal_pot import *

scaling_parameter_c = 128

img = plt.imread('image_cats/A32.jpeg')
img_A = np.array(img)

img = plt.imread('image_cats/B32.jpeg')
img_B = np.array(img)

img = plt.imread('image_cats/C32.jpeg')
img_C = np.array(img)

img = plt.imread('image_cats/D32.jpeg')
img_D = np.array(img)

print_format_string("SHAPE", 20)
print(img_A.shape)

print_format_string("SCALING FACTOR", 20)
print(scaling_parameter_c)

print_format_string("DISTANCE AB", 20)
dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)

print_format_string("DISTANCE AC", 20)
dist_AC, _ = calculate_bicausal_distance_between_images(img_A, img_C, scaling_parameter_c)
print(dist_AC)

print_format_string("DISTANCE AD", 20)
dist_AD, _ = calculate_bicausal_distance_between_images(img_A, img_D, scaling_parameter_c)
print(dist_AD)


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_bicausal_pot_May27.py
# #######SHAPE########
# (32, 32, 3)
# ###SCALING FACTOR###
# 128
# ####DISTANCE AB#####
# 3.3249195064794006
# ####DISTANCE AC#####
# 2.845637177998202
# ####DISTANCE AD#####
# 7.519840021360843
#
# Process finished with exit code 0
#
