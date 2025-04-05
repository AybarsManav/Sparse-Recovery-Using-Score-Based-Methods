import os
from PIL import Image
from torchvision import transforms

# Path to the folder containing CelebA images
input_folder = "data/img_align_celeba"  # Folder with original images
output_folder = "data/processed_celeba"  # Folder to save processed images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define transformations for the images
transform = transforms.Compose([
    transforms.CenterCrop(140),     # Center crop to 140x140
    transforms.Resize((64, 64)),    # Resize cropped images to 64x64
])

# Process and save each image
for image_file in os.listdir(input_folder):
    if image_file.endswith(('.jpg', '.png')):  # Process only image files
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Open the image
        image = Image.open(input_path).convert("RGB")  # Ensure 3-channel RGB

        # Apply transformations
        processed_image = transform(image)

        # Save the processed image
        processed_image.save(output_path)

print(f"Processed images saved to: {output_folder}")