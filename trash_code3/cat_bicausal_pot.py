import matplotlib.pyplot as plt
from distance_bicausal_pot import *
from utils import *

def downsample_image(image, factor):
    # Downsample by slicing
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

down_factor = 32
scaling_parameter_c = 128

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

img = plt.imread('../image_cats/D.jpg')
img_D = np.array(img)
img_D = downsample_image(img_D, down_factor)

print("SHAPE###################################")
print(img_A.shape)

print("scaling factor##########################")
print(scaling_parameter_c)

print("################### Causal Distance Between AB #########################")
dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")


print("################### Causal Distance Between AD #########################")
dist_AD, _ = calculate_bicausal_distance_between_images(img_A, img_D, scaling_parameter_c)
print(dist_AD)
print("#########################################################################")


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_bicausal_pot.py
# SHAPE###################################
# (32, 32, 3)
# scaling factor##########################
# 100
# ################### Causal Distance Between AB #########################
# 2.3870907330685536
# ########################################################################
# ################### Causal Distance Between AD #########################
# 5.964331711155179
# #########################################################################
#
# Process finished with exit code 0


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_bicausal_pot.py
# SHAPE###################################
# (32, 32, 3)
# scaling factor##########################
# 110
# ################### Causal Distance Between AB #########################
# 2.5029948559681214
# ########################################################################
# ################### Causal Distance Between AD #########################
# 6.513630858717156
# #########################################################################
#
# Process finished with exit code 0


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_bicausal_pot.py
# SHAPE###################################
# (32, 32, 3)
# scaling factor##########################
# 128
# ################### Causal Distance Between AB #########################
# 2.7102854687499893
# ########################################################################
# ################### Causal Distance Between AD #########################
# 7.498197377506724
# #########################################################################
#
# Process finished with exit code 0


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_bicausal_pot.py
# DOWNSAMPLE BY SLICING!!!!!!!!!!!!!!!!!
# SHAPE###################################
# (32, 32, 3)
# scaling factor##########################
# 128
# ################### Causal Distance Between AB #########################
# 2.7643088469339543
# ########################################################################
# ################### Causal Distance Between AD #########################
# 7.957364297360058
# #########################################################################
#
# Process finished with exit code 0
