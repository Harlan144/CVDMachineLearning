import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import ResNet50
from functions import *
"""
Run Loaded Weigths from SavesModel0 - 3 to confirm values and generate new graphs.
"""

#This image size worked fine with ResNet50
image_size= (180,180)



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
    x = layers.Dropout(0.5)(x) #Reduce overfitting

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2)
    ]
)

"""
Copied code from classifycvd0.py so it can call model.evaluate on loaded 
"""

model = make_model(input_shape=image_size + (3,), num_classes=2) #Call our model defined above

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

model.load_weights("SavedModelOutputs/SavesModel0/save_at_15.h5") #Call the last save in Model0

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    image_size=image_size,
    shuffle=False
)

# score = model.evaluate(test_ds)[1:] #Return the evaluated metrics
# for i in range(len(score)):
#     print(METRICS[i], score[i])



test_labels = np.concatenate([y for x, y in test_ds], axis=0) 

predictions = model.predict(test_ds)

with open("predictionsModel0","w") as output:

    for i in range(len(predictions)):
        if test_labels[i]==1:
            output.write(f"Unfriendly\t{float(predictions[i])}\n")
        else:
            output.write(f"Friendly\t{float(predictions[i])}\n")

plotPredictions("predictionsModel0","predictionsModel0.png")