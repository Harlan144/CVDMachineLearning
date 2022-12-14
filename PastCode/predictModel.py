import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
#from keras.utils import to_categorical
import matplotlib.pyplot as plt

from functions import plot_cm

image_size= (180,180)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = layers.Rescaling(1./255)(inputs)

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

# Evaluate the model


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    labels= 'inferred',
    image_size=image_size,
    shuffle=False
)


#Not trained:
#model = make_model(input_shape=image_size + (3,) , num_classes=2)
#loss, acc = model.evaluate(test_ds, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


model = make_model(input_shape=image_size + (3,) , num_classes=2)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    #metrics =["acc"]
    #switch for AUC
    metrics=tf.keras.metrics.AUC(),
)

model.load_weights("Saves/save_at_16.h5")




test_labels = np.concatenate([y for x, y in test_ds], axis=0)

predictions = model.predict(test_ds)
converted_predictions = list(map(lambda x: 0 if x<0.5 else 1, predictions))

#con_mat = tf.math.confusion_matrix(labels=y_true, predictions=predictions).numpy()

#largestValue = max(predictions)
#multipliedPredict=  predictions/largestValue
#percentUnfriendly = 47/(388+47)

#sortedPred= sorted(predictions)
#val = sortedPred[-47]
for i in range(len(predictions)):
    print(test_ds.file_paths[i], predictions[i], test_labels[i]) 


plot_cm(test_labels, predictions)

"""
sortedPred= sorted(predictions)
for i in sortedPred[-47:]:
    for index in range(len(predictions)):
        if i == predictions[index]:
            print(i, test_ds.file_paths[index])
            break
"""