import os
import csv

# Paths
image_dir = 'data/training/images'
mask_dir = 'data/training/masks'
csv_path = 'data/pas_training.csv'

# Collect all images and masks
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# Write CSV file
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'mask1_path', 'mask2_path', 'mask3_path', 'mask4_path'])

    for image in images:
        patient_number = image.split('_')[0]
        image_id = image.split('.')[0]  # Extract the base image name without extension
        
        # Find corresponding masks that share the same image prefix
        corresponding_masks = [mask for mask in masks if mask.startswith(image_id)]
        
        # Ensure there are exactly 4 mask paths (if fewer, pad with empty strings)
        mask_paths = [os.path.join(mask_dir, mask) for mask in corresponding_masks]
        while len(mask_paths) < 4:
            mask_paths.append('')

        writer.writerow([os.path.join(image_dir, image)] + mask_paths)