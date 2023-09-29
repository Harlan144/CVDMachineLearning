# -*- coding: utf-8 -*-
"""
CVD Machine Learning Version 0.0

Model includes data augmentation, but not class weighing, transfer learning, or early stopping.
"""

"""
Imports
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import glob
import os
import matplotlib.pyplot as plt

os.listdir("TrainingDataset") #should return unfriendlyCVD and friendlyCVD

image_size = (180, 180) #modify this, this might be too small to work. 
batch_size = 32
seed = 123 #Random seed set for reproducibility. 
validation_split=0.20 

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TrainingDataset/",
    validation_split=validation_split,
    labels='inferred', 
    label_mode='binary', 
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)
#Take 20% of train_ds and withhold as validation through the testing.
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TrainingDataset/",
    validation_split=validation_split,
    labels='inferred',
    label_mode='binary', 
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "TestImages/",
#     image_size=image_size,
#     shuffle=False
# )



#Test to confirm it worked
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off") #1 is notFriendly #0 is friendly
        #plt.savefig("TestImage")


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation
    x = layers.Rescaling(1./255)(x)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) #Reduce overfitting

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,) , num_classes=2) #Create model
model.summary()

epochs = 15 #Run 15 epochs

callbacks = [
    keras.callbacks.ModelCheckpoint("SavesModel0/save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

