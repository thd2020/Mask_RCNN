import os
import csv

# Paths
image_dir = 'images'
mask_dir = 'masks'
csv_path = 'pas.csv'

# Collect all images and masks
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# Write CSV file
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'mask1_path', 'mask2_path', 'mask3_path', 'mask4_path'])

    for image in images:
        patient_number = image.split('_')[0]
        corresponding_masks = [mask for mask in masks if mask.startswith(patient_number)]
        if corresponding_masks:
            writer.writerow([os.path.join(image_dir, image)] + [os.path.join(mask_dir, mask) for mask in corresponding_masks])