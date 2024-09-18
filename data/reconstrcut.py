import os
import shutil
import json
import re

# Paths
root_dir = '/home/lmj/xyx/新勾画部分1/'  # Root directory containing all patients
image_dir = '/home/lmj/xyx/新勾画部分1/images'  # Directory to store all images
mask_dir = '/home/lmj/xyx/新勾画部分1/masks'  # Directory to store all masks (JSONs)

# Ensure target directories exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Function to extract patient number (numbers after the last Chinese character)
def extract_patient_number(patient_folder):
    match = re.search(r'[\u4e00-\u9fff]+([0-9]+)', patient_folder)
    return match.group(1) if match else None

# Loop over patient folders
for patient_folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_folder)
    
    if os.path.isdir(patient_path):
        patient_number = extract_patient_number(patient_folder)  # Extract patient number
        
        if patient_number:
            # Process json files and locate corresponding PNG files using `imagePath` in JSON
            for root, _, files in os.walk(patient_path):
                json_files = [f for f in files if f.endswith('.json')]
                for json_file in json_files:
                    json_path = os.path.join(root, json_file)
                    
                    # Change working directory to the folder where the JSON file is located
                    os.chdir(root)

                    # Copy JSON file
                    new_mask_name = f"{patient_number}_{json_file}"
                    shutil.copy(json_path, os.path.join(mask_dir, new_mask_name))

                    # Read JSON to find the `imagePath`
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract image path from JSON and replace backslashes with forward slashes for Linux compatibility
                    image_relative_path = data.get('imagePath', None).replace('\\', '/')
                    
                    # Use the relative path directly as the source for the image
                    image_path = os.path.normpath(image_relative_path)
                    image_file_name = os.path.basename(image_relative_path)

                    # Ensure the image exists and copy it
                    if os.path.exists(image_path):
                        new_image_name = f"{patient_number}_{image_file_name}"
                        shutil.copy(image_path, os.path.join(image_dir, new_image_name))
                    else:
                        print(f"Image not found: {image_path}")

                    # Change working directory back to the original root
                    os.chdir(root_dir)
