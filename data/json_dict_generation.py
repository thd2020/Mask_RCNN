import os
import json

# Paths to training and testing directories
train_image_dir = 'data/training/images'
train_mask_dir = 'data/training/masks'
test_image_dir = 'data/testing/images'
test_mask_dir = 'data/testing/masks'

# Output paths for the JSON files
train_json_path = 'data/train_mri_segmentation.json'
test_json_path = 'data/test_mri_segmentation.json'

# Define expected mask labels
mask_labels = ['bladder', 'placenta', 'placenta_accreta', 'uterine_myometrium']

# Function to create the dictionary for JSON
def create_image_mask_dict(image_dir, mask_dir):
    image_mask_dict = {}
    
    # Loop over images and find corresponding masks
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            image_base = os.path.splitext(image_file)[0]
            corresponding_masks = []
            
            # Collect the masks corresponding to this image
            for label in mask_labels:
                mask_file = f"{image_base}_{label}.png"
                mask_path = os.path.join(mask_dir, mask_file)
                
                # Check if mask exists and add it to the list
                if os.path.exists(mask_path):
                    corresponding_masks.append(mask_path)
            
            # Ensure that all masks are present
            if len(corresponding_masks) == len(mask_labels):
                image_path = os.path.join(image_dir, image_file)
                image_mask_dict[image_path] = corresponding_masks
    
    return image_mask_dict

# Create JSON data for training set
train_image_mask_dict = create_image_mask_dict(train_image_dir, train_mask_dir)

# Create JSON data for testing set
test_image_mask_dict = create_image_mask_dict(test_image_dir, test_mask_dir)

# Write training data to JSON
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_image_mask_dict, f, ensure_ascii=False, indent=4)

# Write testing data to JSON
with open(test_json_path, 'w', encoding='utf-8') as f:
    json.dump(test_image_mask_dict, f, ensure_ascii=False, indent=4)

print(f"JSON files created: {train_json_path}, {test_json_path}")