import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from functions import plot_cm
import tensorflowjs as tfjs

image_size= (224, 224)


base_model = MobileNetV2(input_shape=image_size+(3,),
                        include_top=False,
                        weights='imagenet')

base_model.trainable = False

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
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

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2), #need more augmenting factors 
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = keras.Input(shape=input_shape)

    # Image augmentation block
    x = data_augmentation(inputs) #apply augmentation

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias)


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


model.load_weights("Saves/first_save.h5")

#ltfjs.converters.save_keras_model(model,"WebsiteStuff")

score = model.evaluate(test_ds)[1:]
for i in range(len(score)):
    print(METRICS[i], score[i])

test_labels = np.concatenate([y for x, y in test_ds], axis=0)

predictions = model.predict(test_ds)
for i in range(len(test_labels)):
    print(test_ds.file_paths[i], test_labels[i], predictions[i])

plot_cm(test_labels, predictions)


