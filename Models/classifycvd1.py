# -*- coding: utf-8 -*-
"""
CVD Machine Learning 1.0
Model includes data augmentation and class weighing. We do not use transfer learning or early stopping.
An initial bias in the model is not used.
"""

"""
Imports
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import matplotlib.pyplot as plt

os.listdir("TrainingDataset") #should return unfriendlyCVD and friendlyCVD

image_size = (180, 180)
batch_size = 32
seed = 123
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
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    image_size=image_size,
    shuffle=False
)


class_weight = {0:0, 1:0}
y = np.concatenate([y for x, y in train_ds], axis=0)
total = 0
for i in y:
    if int(i[0]) in class_weight:
        class_weight[int(i[0])]+=1
        total+=1
    else:
        print("Error:",i)
class_weight[0]=class_weight[0]/total
class_weight[1]=class_weight[1]/total
print(class_weight) #Check the class weights. We have far more Friendly images than unfriendly, so class_weight[0] should be greater.

#initial_bias = np.log([class_weight[1]/class_weight[0]])
#print(initial_bias) #Check our initial bias based on our inbalanced classes.



data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ]
)

def make_model(input_shape):
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
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,))
model.summary()

epochs = 15

callbacks = [
    keras.callbacks.ModelCheckpoint("SavesModel1/save_at_{epoch}.h5"), 
]

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=METRICS,
)

model.fit(
    train_ds, 
    epochs=epochs, 
    callbacks=callbacks, 
    validation_data=val_ds,
    class_weight=class_weight #Call the class weights here to get more balanced results. Otherwise, the model predicts everything
    # to be "Friendly".
)


