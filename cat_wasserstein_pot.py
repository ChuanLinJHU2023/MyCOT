import matplotlib.pyplot as plt
from distance_wasserstein_pot import *
from utils import *

def downsample_image(image, factor):
    # Downsample by slicing
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

down_factor = 32
scaling_parameter_c = 128

img = plt.imread('cat_images/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('cat_images/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

img = plt.imread('cat_images/D.jpg')
img_D = np.array(img)
img_D = downsample_image(img_D, down_factor)


print("################################SHAPE################################")
print(img_A.shape)

print("################################SCALING FACTOR################################")
print(scaling_parameter_c)

print("################### Wasser Distance Between AB #########################")
dist_AB, _ = calculate_wasserstein_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")


print("################### Wasser Distance Between AD #########################")
dist_AD, _ = calculate_wasserstein_distance_between_images(img_A, img_D, scaling_parameter_c)
print(dist_AD)
print("########################################################################")


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_wasserstein_pot.py
# ################################SHAPE################################
# (32, 32, 3)
# ################################SCALING FACTOR################################
# 128
# ################### Wasser Distance Between AB #########################
# 2.447481574719718
# ########################################################################
# ################### Wasser Distance Between AD #########################
# 0.7104200898785274
# ########################################################################
#
# Process finished with exit code 0





# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyCOT/cat_wasserstein_pot.py
# SLICE TO DOWN SAMPLE!!!!!!!!!!!
# ################################SHAPE################################
# (32, 32, 3)
# ################################SCALING FACTOR################################
# 128
# ################### Wasser Distance Between AB #########################
# 2.3849868542628903
# ########################################################################
# ################### Wasser Distance Between AD #########################
# 0.5726713952781953
# ########################################################################
#
# Process finished with exit code 0
