import numpy as np
from PIL import Image
from main import *


image_path1 = 'CMNIST/train/0/1-5.png'
img1 = Image.open(image_path1)
img_array1 = np.array(img1)

image_path2 = 'CMNIST/train/1/8-4.png'
img2 = Image.open(image_path2)
img_array2 = np.array(img2)

print("The OT distance and COT distance between the following two images are as follows")
print(image_path1)
print(image_path2)
print(calculate_OT_distance_for_image(img_array1, img_array2, distance_between_channel=5))
print(calculate_COT_distance_for_image(img_array1, img_array2, distance_between_channel=5))
print("#################################################################################")



image_path1 = 'CMNIST/train/0/1-5.png'
img1 = Image.open(image_path1)
img_array1 = np.array(img1)

image_path2 = 'CMNIST/train/0/1-4.png'
img2 = Image.open(image_path2)
img_array2 = np.array(img2)

print("The OT distance and COT distance between the following two images are as follows")
print(image_path1)
print(image_path2)
print(calculate_OT_distance_for_image(img_array1, img_array2, distance_between_channel=5))
print(calculate_COT_distance_for_image(img_array1, img_array2, distance_between_channel=5))
print("#################################################################################")


# /opt/anaconda3/envs/MyProject1/bin/python /Users/chuanlin/JHU-LAB/MyProject1/digits.py
# The OT distance and COT distance between the following two images are as follows
# CMNIST/train/0/1-5.png
# CMNIST/train/1/8-4.png
# 1.248071123841414
# 1.4315662194251364
# #################################################################################
# The OT distance and COT distance between the following two images are as follows
# CMNIST/train/0/1-5.png
# CMNIST/train/0/1-4.png
# 2.695767911178179
# 2.6957679111781845
# #################################################################################
#
# Process finished with exit code 0