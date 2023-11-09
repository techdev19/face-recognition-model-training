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


# for images, labels in train_ds.take(1):
#    for i in range(6):
#        ax = plt.subplot(2, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")


from keras import layers

# normalization
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# flipping and rotation
flip_layer = layers.RandomFlip("horizontal_and_vertical")
rotation_layer = layers.RandomRotation(0.2)
augment_layers = keras.keras.Sequential([flip_layer, rotation_layer]) # it will be added directly to the model later
augmented_ds = train_ds.map(lambda x, y: (augment_layers(x), y))


train_ds = train_ds.concatenate(augmented_ds)



#In the keras_vggface/models.py file, change the import from
# keras.engine.topology import get_source_inputs
#to: from keras.utils.layer_utils import get_source_inputs

from keras_vggface.vggface import VGGFace

base_model = VGGFace(model='vgg16',
    include_top=False, # load only feature extraction layers
    input_shape=(224, 224, 3))
base_model.summary()


nb_class = len(class_names)

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
out = Dense(nb_class, activation='softmax')(x)


# new model
model = Model(inputs = base_model.input, outputs = out)
model.summary()

# not training the first 19 layers 
for layer in model.layers[:len(base_model.layers)]:
    layer.trainable = False

# train the rest
for layer in model.layers[len(base_model.layers):]:
    layer.trainable = True




#optional

AUTOTUNE = keras.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model.compile(optimizer='Adam',
    # if output is One hot encoded (ie: [0, 0, 1, 0]) 
    # use CategoricalCrossentropy
    loss= keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

model.fit(
    train_ds,
    batch_size = 1,
    validation_data=val_ds,
    verbose = 1,
    epochs = 10
)



# save model
model.save('models/vgg16_10epoch_face_cnn_model.h5')

# save label
import pickle
class_dictionary = {
    index: value for (index, value) in enumerate(class_names)
}
face_label_filename = 'models/vgg16_10epoch_face_label.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)


