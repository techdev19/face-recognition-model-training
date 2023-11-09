# load model
from keras.models import load_model
model = load_model('models/vgg16_10epoch_face_cnn_model.h5')

# load label
import pickle
face_label_filename = 'models/vgg16_10epoch_face_label.pickle'
with open(face_label_filename, "rb") as f: class_dictionary = pickle.load(f)
class_list = [value for _, value in class_dictionary.items()]
print(class_list)


import cv2
imagePath = 'Brad Pitt/003_7a6b2156.jpg'
image = cv2.imread(imagePath)

import dlib
import matplotlib.pyplot as plt
import numpy as np

from dlib_extractor import *

# detect and extract face from the image
faces = detectFace(image)
face = faces[0]
face = cv2.resize(face, (224, 224))


from keras.preprocessing import image
from keras_vggface import utils

face = image.img_to_array(face)
face = np.expand_dims(face, axis=0)
face = utils.preprocess_input(face, version=1)

preds = model.predict(face)
print(preds)
print("Predicted face: " + class_list[preds[0].argmax()])
print("============================\n")