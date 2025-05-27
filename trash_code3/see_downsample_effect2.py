import matplotlib.pyplot as plt
from distance_bicausal_pot import *
from utils import *

down_factor = 16
scaling_parameter_c = 4

print("################### First Downsample Method #########################")
def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.array(downsampled_image)

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
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

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
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

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")



print("################### Fourth Downsample Method #########################")
def downsample_image(image, factor):
    # Downsample by slicing
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

img = plt.imread('../image_cats/A.jpg')
img_A = np.array(img)
img_A = downsample_image(img_A, down_factor)

img = plt.imread('../image_cats/B.jpg')
img_B = np.array(img)
img_B = downsample_image(img_B, down_factor)

dist_AB, _ = calculate_bicausal_distance_between_images(img_A, img_B, scaling_parameter_c)
print(dist_AB)
print("########################################################################")

