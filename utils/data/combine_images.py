from PIL import Image
import os

# Define the folder path containing the images
folder_path = '../data/67_ray/current_image1/'

# Get a list of image filenames in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Create a list to store individual image objects
images = []

# Open and append each image to the 'images' list
for image_file in image_files:
    image = Image.open(os.path.join(folder_path, image_file))
    images.append(image)

# Get the dimensions of the first image (assuming all images have the same dimensions)
width, height = images[0].size

# Set the number of images per row
images_per_row = 10

# Calculate the number of rows
num_rows = len(images) // images_per_row + (len(images) % images_per_row > 0)

# Create a new blank image with the combined size
combined_width = width * images_per_row
combined_height = height * num_rows
combined_image = Image.new('RGB', (combined_width, combined_height))

# Paste each image into the combined image, arranging them in rows
for i, image in enumerate(images):
    row = i // images_per_row
    col = i % images_per_row
    combined_image.paste(image, (col * width, row * height))

# Save the combined image
combined_image.save('./combined_image.png')
