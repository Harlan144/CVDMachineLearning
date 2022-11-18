import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from functions import plot_cm

image_size= (224, 224)


base_model = MobileNetV2(input_shape=image_size+(3,),
                        include_top=False,
                        weights='imagenet')

base_model.trainable = False

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    labels='inferred',
    image_size=image_size,
    shuffle=False
)

class_weight = {0:0, 1:0}
y = np.concatenate([y for x, y in test_ds], axis=0)
total = 0
for i in y:
    if int(i) in class_weight:
        class_weight[int(i)]+=1
        total+=1
    else:
        print("Error:",i)

initial_bias = np.log([class_weight[1]/class_weight[0]])

print(class_weight)


def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)



model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    #metrics =["acc"]
    #switch for AUC
    metrics=tf.keras.metrics.AUC(),
)

model.load_weights("Saves/last_save.h5")

test_labels = np.concatenate([y for x, y in test_ds], axis=0)

predictions = model.predict(test_ds)
converted_predictions = list(map(lambda x: 0 if x<0.5 else 1, predictions))

for i in range(len(predictions)):
    print(test_ds.file_paths[i], predictions[i], test_labels[i]) 


plot_cm(test_labels, predictions)
