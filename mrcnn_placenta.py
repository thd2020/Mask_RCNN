#%% md
# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.
#%%
from google.colab import drive
drive.mount("/content/drive")
#%%
!git clone https://github.com/lrpalmer27/Mask-RCNN-TF2 /content/drive/MyDrive/mask_rcnn
#%%
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

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append("/content/drive/MyDrive/mask_rcnn/")  # To find local version of the library
# from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
#%%
sys.path.append(os.path.join(sys.path[0], 'mrcnn'))
#%% md
# ## Configurations
#%%
from mrcnn.config import Config


class ImgsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "placneta"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 1 shape

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ImgsConfig()
config.display()
#%% md
# ## Notebook Preferences
#%%
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
#%% md
# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()
#%%
train_path = "data_demo/image2label_train.json"
train_backup_path = "data_demo/image2label_train_bck.json"
test_path = "data_demo/label2image_test.json"
test_backup_path = "data_demo/image2label_test_bck.json"
# restore test and train json
test_json = open(test_backup_path)
test_dict = json.load(test_json)
train_json = open(train_backup_path)
train_dict = json.load(train_json)
#restore type dict specifically
accreta_dict = {}
increta_dict = {}
normal_dict = {}
type_dict = {}
rc_test_dict = {}
all_dict = {}
for img in train_dict:
  type_dict[img] = img.split(os.path.sep)[2]
for label in test_dict:
  img = test_dict[label]
  type_dict[img] = img.split(os.path.sep)[2]
  if (type_dict[img] == "accreta"):
    accreta_dict[label] = img
  elif (type_dict[img] == "increta"):
    increta_dict[label] = img
  elif (type_dict[img] == "normal"):
    normal_dict[label] = img
#spetial to reverse
for k, v in test_dict.items():
  rc_test_dict[v] = []
  rc_test_dict[v].append(k)
all_dict = {**train_dict, **rc_test_dict}
#%%
from itertools import islice
import cv2
def nth_key(dct, n):
    it = iter(dct)
    # Consume n elements.
    next(islice(it, n, n), None)
    # Return the value at the current position.
    # This raises StopIteration if n is beyond the limits.
    # Use next(it, None) to suppress that exception.
    return next(it)
#%%
nth_key(train_dict, 328)
#%%
np.bool = np.bool_
class ImgsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_images(self, source, x_dict, *ts):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        source = "placenta"
        # Add classes
        for i in range(len(ts)):
          self.add_class(source, i+1, ts[i])

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(len(x_dict)):
            img = nth_key(x_dict, i)
            self.add_image(source=source, image_id=i, path=img,
                          placenta_type = type_dict[img])

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "placenta":
            return info["placenta_type"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, img):
        """Generate instance masks for shapes of the given image ID.
        """
        mask = cv2.imread(all_dict[img][0])
        class_ids = np.array([self.class_names.index(type_dict[img])])
        return mask.astype(np.bool), class_ids.astype(np.int32)
#%%
# Training dataset
dataset_train = ImgsDataset()
dataset_train.load_images("placenta", train_dict, "normal", "accreta", "increta")
dataset_train.prepare()

# Validation dataset
dataset_val = ImgsDataset()
dataset_val.load_images("placenta", rc_test_dict, "normal", "accreta", "increta")
dataset_val.prepare()
#%%
# Load and display random samples
imgs = np.random.choice(list(train_dict.keys()), 4)
for img in imgs:
    mask, class_ids = dataset_train.load_mask(img)
    img = cv2.imread(img)
    visualize.display_top_masks(img, mask, class_ids, dataset_train.class_names)
#%%
# Load and display random samples
imgs = np.random.choice(list(rc_test_dict.keys()), 4)
for img in imgs:
    mask, class_ids = dataset_val.load_mask(img)
    img = cv2.imread(img)
    visualize.display_top_masks(img, mask, class_ids, dataset_val.class_names)
#%% md
# ## Create Model
#%%
!git remote set-url origin https://gitclone.com/github.com/leekunhee/Mask_RCNN.git
!git pull origin master:main
#%%
!git reset --hard origin/master
!git pull origin master:master
#%%
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
#%%
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

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
    model.load_weights(model.find_last(), by_name=True)
#%% md
# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.
#%%
print("Train items: ", len(train_dict), "; Val items: ", len(rc_test_dict), "; all itmes: ", len(all_dict))
#%%
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')
#%%
# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")
#%%
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)
#%% md
# ## Detection
#%%
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

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
model.load_weights(model_path, by_name=True)
#%%
# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))
#%%
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())
#%% md
# ## Evaluation
#%%
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
#%%
