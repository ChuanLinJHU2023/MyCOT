import os
import random
from PIL import Image, ImageOps

# Paths
original_train_dir = 'MNIST/train'
original_test_dir = 'MNIST/test'
new_train_dir = 'image_digits/train'
new_test_dir = 'image_digits/test'

# Create new directory structure
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

# Define some color options
foreground_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
background_colors = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 128, 128), (128, 0, 128)]

def create_color_variant(image_path, save_path):
    img = Image.open(image_path).convert('L')  # Grayscale

    # Threshold to get binary mask
    threshold = 128
    binary_mask = img.point(lambda p: p > threshold and 255)

    # Invert mask for background
    background_mask = ImageOps.invert(binary_mask)

    # Randomly select colors
    fg_color = random.choice(foreground_colors)
    bg_color = random.choice(background_colors)

    # Create color images
    fg_img = Image.new('RGB', img.size, color=fg_color)
    bg_img = Image.new('RGB', img.size, color=bg_color)

    # Composite foreground and background using masks
    colored_img = Image.composite(fg_img, bg_img, binary_mask)

    # Save the result
    colored_img.save(save_path)

def process_directory(source_dir, target_dir):
    for label in os.listdir(source_dir):
        label_source_path = os.path.join(source_dir, label)
        label_target_path = os.path.join(target_dir, label)
        if not os.path.isdir(label_source_path):
            continue
        os.makedirs(label_target_path, exist_ok=True)
        for filename in os.listdir(label_source_path):
            if filename.endswith('.png'):
                original_image_path = os.path.join(label_source_path, filename)
                # Generate 5 diverse color variants
                for i in range(1, 6):
                    new_filename = filename.replace('.png', f'-{i}.png')
                    save_path = os.path.join(label_target_path, new_filename)
                    create_color_variant(original_image_path, save_path)

# Process train and test sets
process_directory(original_train_dir, new_train_dir)
process_directory(original_test_dir, new_test_dir)

print("Diverse color variants created in 'image_digits/train/' and 'image_digits/test/'.")