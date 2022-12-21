"""
Run Loaded Weigths from SavesModel5, SavesModel6 or SavesModel7 to confirm values and generate new graphs.
Uncomment tfjs.converters.save_keras_model(model,"ModelAsJSON")  to export to our website as a JSON.
"""


import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from functions import plot_cm
import tensorflowjs as tfjs


#This image size was found to work well with MobileNetV2
image_size= (224, 224)


"""
Copied code from classifycvd7.py so it can call model.evaluate on loaded 
"""
base_model = MobileNetV2(input_shape=image_size+(3,),
                        include_top=False,
                        weights='imagenet')

base_model.trainable = False

#Only test on our test_ds
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    image_size=image_size,
    shuffle=False
)

#Weigh the classes (friendly or not friendly) according to their prevelance in our test dataset
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
        layers.RandomRotation(0.2)
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias) #decreases loss in first few epochs.
    inputs = keras.Input(shape=input_shape) 

    x = data_augmentation(inputs) #apply augmentation from above

    x = base_model(x, training=False) #Call MobileNetV2

    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.3)(x) #decreases overfitting

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x) #returns value between 0 and 1. 1 is unfriendly.
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias) #Calls the model.


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


#
model.load_weights("SavesModel7/first_save.h5") #Call the save from Model 7.

#If you want to export the file to be run in a webserver, uncomment the code below.
#tfjs.converters.save_keras_model(model,"ModelAsJSON") 

score = model.evaluate(test_ds)[1:]
for i in range(len(score)):
    print(METRICS[i], score[i]) 

test_labels = np.concatenate([y for x, y in test_ds], axis=0) #list of labels in test_ds

predictions = model.predict(test_ds) #returns array of predictions
for i in range(len(test_labels)):
    print(test_ds.file_paths[i], test_labels[i], predictions[i])

plot_cm(test_labels, predictions, p=0.5, savePath="SavesModel7/PlotConfusionMatrix") #plot the confusion matrix.


