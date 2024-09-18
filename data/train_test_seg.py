import os
import shutil
import random

# Paths
root_dir = './data'  # Root directory containing images and masks
image_dir = os.path.join(root_dir, 'images')
mask_dir = os.path.join(root_dir, 'masks')
train_dir = './data/training'
test_dir = './data/testing'

# Directories for training and testing sets
train_images_dir = os.path.join(train_dir, 'images')
train_masks_dir = os.path.join(train_dir, 'masks')
test_images_dir = os.path.join(test_dir, 'images')
test_masks_dir = os.path.join(test_dir, 'masks')

# Ensure directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# Group images and masks by patient ID
patient_images = {}
patient_masks = {}

# Function to extract patient ID from filename (e.g., "20220513003495" from "20220513003495_24_23.png")
def extract_patient_id(filename):
    return filename.split('_')[0]

# Organize images by patient ID
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png'):
        patient_id = extract_patient_id(image_file)
        if patient_id not in patient_images:
            patient_images[patient_id] = []
        patient_images[patient_id].append(image_file)

# Organize masks by patient ID
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.png'):
        patient_id = extract_patient_id(mask_file)
        if patient_id not in patient_masks:
            patient_masks[patient_id] = []
        patient_masks[patient_id].append(mask_file)

# Get list of patient IDs
patient_ids = list(patient_images.keys())

# Randomly select 4 patients for training and 2 for testing
random.shuffle(patient_ids)
train_patients = patient_ids[:4]
test_patients = patient_ids[4:]

# Function to copy files for a given patient ID
def copy_files(patient_id, dest_images_dir, dest_masks_dir):
    # Copy images
    for image_file in patient_images.get(patient_id, []):
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(dest_images_dir, image_file))
    
    # Copy masks
    for mask_file in patient_masks.get(patient_id, []):
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(dest_masks_dir, mask_file))

# Copy training data
for patient_id in train_patients:
    print(f'Copying training data for patient: {patient_id}')
    copy_files(patient_id, train_images_dir, train_masks_dir)

# Copy testing data
for patient_id in test_patients:
    print(f'Copying testing data for patient: {patient_id}')
    copy_files(patient_id, test_images_dir, test_masks_dir)

print("Data split by patient completed.")
