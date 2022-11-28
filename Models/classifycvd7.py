# -*- coding: utf-8 -*-
"""
CVD Machine Learning 7.0
The model works the same as classifiycvd6.py. Added code to plot more metrics. Verified that our random seed produces the same results.
Model includes class weighing, early stoppping, data augmentation and transfer learning from MobileNetV2.
Attempted fine tuning. An initial bias based on class weights is used.
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
from functions import *
from tensorflow.keras.applications import MobileNetV2
from keras.models import Model


#This image size was found to work well with MobileNetV2
image_size = (224, 224) 
batch_size = 32
seed = 123 #Set our random seed as 123 to ensure reproducibility
validation_split=0.20 

#Used MobileNetV2 as our base model. Initially, the base_model's layers are not set as trainable.
base_model = MobileNetV2(input_shape=image_size+(3,),
                        include_top=False,
                        weights='imagenet')

base_model.trainable = False

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TrainingDataset/",
    validation_split=validation_split,
    labels='inferred', #might not need this line
    label_mode='binary',
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)
#Withhold 20% of our train_ds as validation while the model trains.
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TrainingDataset/",
    validation_split=validation_split,
    labels='inferred', #might not need this line
    label_mode='binary',
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)

#Used for final confirmation after the model has been trained.
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "TestImages/",
    image_size=image_size,
    shuffle=False
)

#Return labels from our test_ds. Used later.
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

class_weight = {0:0, 1:0}
y = np.concatenate([y for x, y in train_ds], axis=0)
total = 0
for i in y:
    if int(i[0]) in class_weight:
        class_weight[int(i[0])]+=1
        total+=1
    else:
        print("Error:", i)

#Sets intial_bias to be used while training the model. Decreases loss during the first few epochs.
initial_bias = np.log([class_weight[1]/class_weight[0]])
print("Initial Bias: ", initial_bias) 

#Sets class weights. 0 = Friendly. 1 = Unfriendly. We have unbalanced classes, so this is needed to give more accurate results.
class_weight[0]=(1/class_weight[0])*(total/2)
class_weight[1]=(1/class_weight[1])*(total/2)


#0 is weighed less since Friendly images are most frequent.
print('Weight for friendly, class 0: {:.2f}'.format(class_weight[0])) 
#1 is weighed more since Unfriendly images are less frequent.
print('Weight for unfriendly, class 1: {:.2f}'.format(class_weight[1])) 

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2)
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias) #Use the initial_bias as defined above. Decreases initial loss.
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = base_model(x, training=False) #Call our MobileNetV2 base_model.

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) #Used to decrease overfitting

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias) #Call our model as defined above.
model.summary() #Print a summary of the model onto the command line.

epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint("SavesModel7/save_at_{epoch}.h5"),
    keras.callbacks.EarlyStopping(
    monitor='val_prc', #Stops when validation precision is the highest.
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
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

#Call the model. Use defined class weights to improve accuracy.

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    class_weight=class_weight
)

#Calls function in functions.py. Save a plot of precision, recall, PRC and loss through the epochs
plot_metrics(history, "SavesModel7/history")


#Write out the saved metrics to a new file.
with open("SavesModel7/Evaluated", "w") as file:
     results = model.evaluate(test_ds)
     for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")


#Save the current model as first_save.h5. This can be accessed through saveModel5.py.

model.save("SavesModel7/first_save.h5")


#Plot a confusion matrix of the current model.
predictions = model.predict(test_ds)
plot_cm(test_labels, predictions, p=0.5, savePath="SavesModel7/ConfusionMatrix1")


#Fine tune the model by setting the base_model as trainable.
base_model.trainable = True

#Lower the learning rate significantly since the base model is far bigger than our model.
model.compile(optimizer=keras.optimizers.Adam(1e-5),  
              loss="binary_crossentropy",
              metrics=METRICS)

new_callbacks = [
    keras.callbacks.EarlyStopping(
    verbose=1,
    patience=10,
    restore_best_weights=True)
]

history1 = model.fit(train_ds, epochs=15,validation_data=val_ds, callbacks= new_callbacks)


#Freeze all layers of our model so that we can save it.
base_model.trainable = False
def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

model_freezed = freeze_layers(model)

#Save our current model that has been fine tuned at last_save.h5.
model_freezed.save('SavesModel7/last_save.h5')

#Calls function in functions.py. Save a plot of precision, recall, PRC and loss through the epochs
plot_metrics(history1, "SavesModel7/history1")

with open("SavesModel7/EvaluatedFineTuning", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")

newPredict = model.predict(test_ds)
plot_cm(test_labels, newPredict, p=0.5, savePath="SavesModel7/ConfusionMatrix2")