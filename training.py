import os
import pandas as pd
import numpy as np

import tensorflow as keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory


batch_size = 32
img_height = 224
img_width = 224

processed_folder_name = 'processed'
processed_folder_dir = os.path.join(".", processed_folder_name)

train_ds = image_dataset_from_directory(
  processed_folder_name,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = image_dataset_from_directory(
  processed_folder_name,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


