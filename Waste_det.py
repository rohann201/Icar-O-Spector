import os
import sys
import random
import math
import re
import time
import glob
import skimage
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import math

ROOT_DIR = os.getcwd()
print(ROOT_DIR)

sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from google.colab.patches import cv2_imshow
from trash import trash

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

TRASH_WEIGHTS_PATH = "/content/drive/MyDrive/Copy of mask_rcnn_trash_0200_030519_large.h5" #the best

print('Weights being used: ', TRASH_WEIGHTS_PATH)

config = trash.TrashConfig()
TRASH_DIR = 'trash'
TRASH_DIR
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/gpu:0"
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

dataset = trash.TrashDataset()
dataset.load_trash(TRASH_DIR, "val")

dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
weights_path = os.path.join(ROOT_DIR, TRASH_WEIGHTS_PATH)
model.load_weights(weights_path, by_name=True)
print("Loading weights ", TRASH_WEIGHTS_PATH)
jpg = glob.glob("images/*.jpg")
jpeg = glob.glob("images/*.jpeg")
jpg.extend(jpeg)

for image in jpg:
    print(image)
    image = skimage.io.imread('{}'.format(image))

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    b,a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")

print(b)

a = cv2.cvtColor(np.float32(a), cv2.COLOR_BGR2RGB)
dim = (a.shape[0] * a.shape[1])
print("Percentage of image containing waste: ", b/dim * 100)

count = 2 #placeholder as of now, final implementation explained in future prospect
count_score = 50*count/15
per = b/dim*100
per = per/2
sum = count_score + per
print(math.ceil(sum))
