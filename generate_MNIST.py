import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define directories
output_dir = "MNIST"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load MNIST dataset
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()

# Function to save images
def save_images(images, labels, folder):
    for i, (img, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(folder, str(label))
        os.makedirs(label_dir, exist_ok=True)
        filename = os.path.join(label_dir, f"{i}.png")
        plt.imsave(filename, img, cmap='gray')

# Save training images
save_images(mnist_x_train, mnist_y_train, train_dir)
# Save testing images
save_images(mnist_x_test, mnist_y_test, test_dir)

print("MNIST dataset saved as PNG images in directory 'MNIST'.")