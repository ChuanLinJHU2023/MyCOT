import matplotlib.pyplot as plt
import cv2
import numpy as np
from main import *

def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.array(downsampled_image)


down_factor = 32
distance_between_channel = 5
img = plt.imread('../cat_images/A.jpg')
img_A = np.array(img)
img_A_down = downsample_image(img_A, down_factor)

img = plt.imread('../cat_images/B.jpg')
img_B = np.array(img)
img_B_down = downsample_image(img_B, down_factor)

img = plt.imread('../cat_images/D.jpg')
img_D = np.array(img)
img_D_down = downsample_image(img_D, down_factor)



print("The OT distance and COT distance between the following two images are as follows")
print("CatA")
print("CatB")
print(calculate_OT_distance_for_image(img_A_down, img_B_down, distance_between_channel=distance_between_channel))
print(calculate_COT_distance_for_image(img_A_down, img_B_down, distance_between_channel=distance_between_channel))
print("#################################################################################")
print("The OT distance and COT distance between the following two images are as follows")
print("CatA")
print("CatD")
print(calculate_OT_distance_for_image(img_A_down, img_D_down, distance_between_channel=distance_between_channel))
print(calculate_COT_distance_for_image(img_A_down, img_D_down, distance_between_channel=distance_between_channel))
print("#################################################################################")


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyProject1/cat.py
# The OT distance and COT distance between the following two images are as follows
# CatA
# CatB
# 2.920486002565117
# 3.0467042643302658
# #################################################################################
# The OT distance and COT distance between the following two images are as follows
# CatA
# CatD
# 0.9369642122709922
# 2.068981839882437
# #################################################################################
#
# Process finished with exit code 0
