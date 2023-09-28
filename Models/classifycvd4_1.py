# -*- coding: utf-8 -*-
"""
CVD Machine Learning 4.1 (ResNet50)
Model includes class weighing, early stoppping, data augmentation and transfer learning from MobileNetV2.
Attempted fine tuning.
An initial bias based on class weights is used.
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
from functions import *
from tensorflow.keras.applications import MobileNetV2
from keras.models import Model



image_size = (224, 224)
batch_size = 32
seed = 123
validation_split=0.20 

#Used transfer learning from MobileNetV2 to build off of as our base model.
base_model = MobileNetV2(input_shape=image_size+(3,),
                        include_top=False,
                        weights='imagenet')

base_model.trainable = False

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

initial_bias = np.log([class_weight[1]/class_weight[0]])
print("Initial Bias: ", initial_bias)

class_weight[0]=(1/class_weight[0])*(total/2)
class_weight[1]=(1/class_weight[1])*(total/2)

print('Weight for friendly, class 0: {:.2f}'.format(class_weight[0])) 
print('Weight for unfriendly, class 1: {:.2f}'.format(class_weight[1]))


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2)
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) #Same as 7.

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias)
model.summary()

epochs = 30 

callbacks = [
    keras.callbacks.ModelCheckpoint("SavedModelOutputs/SavesModel5/save_at_{epoch}.h5"),
    keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='auto',
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
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=METRICS,
)
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    class_weight=class_weight
)

plot_metrics(history, "SavedModelOutputs/SavesModel4_1/history")

#Evaluate metrics before fine tuning.
with open("SavedModelOutputs/SavesModel4_1/Evaluated", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")



#Save the current model as first_save.h5. This can be accessed through saveModel5.py.

model.save("SavedModelOutputs/SavesModel4_1/first_save.h5")


#Plot a confusion matrix of the current model.
predictions = model.predict(test_ds)
plot_cm(test_labels, predictions, p=0.5, savePath="SavedModelOutputs/SavesModel4_1/ConfusionMatrix1")


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
model_freezed.save('SavedModelOutputs/SavesModel4_1/last_save.h5')

#Calls function in functions.py. Save a plot of precision, recall, PRC and loss through the epochs
plot_metrics(history1, "SavedModelOutputs/SavesModel4_1/history1")

with open("SavedModelOutputs/SavesModel4_1/EvaluatedFineTuning", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")

newPredict = model.predict(test_ds)
plot_cm(test_labels, newPredict, p=0.5, savePath="SavedModelOutputs/SavesModel4_1/ConfusionMatrix2")