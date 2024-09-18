import os
import json
import numpy as np
from PIL import Image
import labelme

# Label translations
label_mapping = {
    "胎盘": "placenta",
    "胎盘植入部分": "placenta_accreta",
    "膀胱": "bladder",
    "子宫肌层": "uterine_myometrium"
}

# Convert JSON to individual mask files
def convert_json_to_masks(json_path, output_dir):
    with open(json_path) as f:
        data = json.load(f)

    image_data = labelme.utils.img_b64_to_arr(data['imageData'])
    h, w = image_data.shape[:2]

    # Create a blank mask for each label
    masks = {label: np.zeros((h, w), dtype=np.uint8) for label in label_mapping.values()}

    for shape in data['shapes']:
        label = shape['label']
        if label in label_mapping:
            points = shape['points']
            mask = labelme.utils.shape_to_mask((h, w), points, shape_type='polygon')
            masks[label_mapping[label]] = np.maximum(masks[label_mapping[label]], mask.astype(np.uint8))

    # Save each mask as PNG
    for label, mask in masks.items():
        mask_name = os.path.basename(json_path).replace('.json', f"_{label}.png")
        Image.fromarray(mask * 255).save(os.path.join(output_dir, mask_name))

# Process all JSON files in a directory
json_dir = '/home/lmj/xyx/新勾画部分1/jsons'
output_mask_dir = '/home/lmj/xyx/新勾画部分1/masks'
os.makedirs(output_mask_dir, exist_ok=True)

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        convert_json_to_masks(os.path.join(json_dir, json_file), output_mask_dir)