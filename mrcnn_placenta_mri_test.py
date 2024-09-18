import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import random as rd
import json
import matplotlib
import matplotlib.pyplot as plt

from samples.shapes.shapes import ShapesConfig

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append("/content/drive/MyDrive/mask_rcnn/")  # To find local version of the library
# from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
sys.path.append(os.path.join(sys.path[0], 'mrcnn'))
from mrcnn.config import Config


class MRIConfig(Config):
    NAME = "mri_segmentation"
    BATCH_SIZE = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # Background + 4 classes (bladder, placenta, etc.)
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # Anchors for MRI scales
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    USE_MINI_MASK = False

config = MRIConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

train_path = "data/train_mri_segmentation.json"
test_path = "data/test_mri_segmentation.json"

with open(train_path, 'r') as f:
    train_dict = json.load(f)

with open(test_path, 'r') as f:
    test_dict = json.load(f)
    

np.bool = np.bool_
class MRI_Dataset(utils.Dataset):
    """Dataset class for MRI dataset."""
    
    def load_images(self, source, x_dict, *ts):
        source = "mri_segmentation"
        for i, label in enumerate(ts):
            self.add_class(source, i+1, label)

        for i, (image_path, mask_paths) in enumerate(x_dict.items()):
            self.add_image(source=source, image_id=i, path=image_path, mask_paths=mask_paths)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "mri_segmentation":
            return info
        else:
            return super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for the given image ID.
        Each image has multiple masks (e.g., bladder, placenta, etc.)
        """
        info = self.image_info[image_id]
        mask_paths = info['mask_paths']  # Paths to all masks (bladder, placenta, etc.)
        
        # Initialize a mask array of the correct size (height, width, number of masks)
        count = len(mask_paths)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        
        # Read and assign each mask to the mask array
        for i, mask_path in enumerate(mask_paths):
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
            mask[:, :, i] = mask_image  # Assign the mask to the correct slice
        
        # Handle occlusions (same logic as the original)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        
        # Map class names to class IDs
        class_ids = np.array([self.class_names.index(label) for label in ['bladder', 'placenta', 'placenta_accreta', 'uterine_myometrium']])
        
        return mask.astype(np.bool_), class_ids.astype(np.int32)

# Training dataset
dataset_train = MRI_Dataset()
dataset_train.load_images("placenta_mri", train_dict, 'bladder', 'placenta', 'placenta_accreta', 'uterine_myometrium')
dataset_train.prepare()

# Validation dataset
dataset_val = MRI_Dataset()
dataset_val.load_images("placenta_mri", test_dict, 'bladder', 'placenta', 'placenta_accreta', 'uterine_myometrium')
dataset_val.prepare()

# Load and display random samples
# Load and display random samples using image IDs
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     mask, class_ids = dataset_train.load_mask(image_id)
#     img = dataset_train.load_image(image_id)[0]  # Load the actual image using the image ID
#     visualize.display_top_masks(img, mask, class_ids, dataset_train.class_names)

# # Load and display random samples
# image_ids = np.random.choice(dataset_val.image_ids, 4)
# for image_id in image_ids:
#     mask, class_ids = dataset_train.load_mask(image_id)
#     img = dataset_train.load_image(image_id)[0]  # Load the actual image using the image ID
#     visualize.display_top_masks(img, mask, class_ids, dataset_val.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

class InferenceConfig(MRIConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
inference_config.display()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

for tests in range(0, 5):
    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                            image_id)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    print('model_path: ', model_path)
    example_path = os.path.join(config.CHECKPOINT_PATH, "examples")
    if not os.path.exists(example_path):
        os.makedirs(example_path)

    gt_plt = visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))
    gt_plt_path=os.path.join(config.CHECKPOINT_PATH, "examples", str(image_id)+"gt")
    gt_plt.savefig(gt_plt_path)

    results = model.detect([original_image], verbose=1)

    r = results[0]
    test_plt = visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())
    test_plt_path=os.path.join(config.CHECKPOINT_PATH, "examples", str(image_id)+"prediction")
    test_plt.savefig(test_plt_path)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
Precisions = []
Overlaps = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    Precisions.append(np.mean(precisions))
    Overlaps.append(np.mean(overlaps))

print("mAP: ", np.mean(APs))
print("mPrecision: ", np.mean(Precisions))
print("mOverlap: ", np.mean(Overlaps))

