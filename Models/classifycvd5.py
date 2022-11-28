# -*- coding: utf-8 -*-
"""
CVD Machine Learning 5.0 (MobileNetV2)
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
        layers.RandomRotation(0.1)
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) #Decreased Dropout rate even further to 0.2.

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias)
model.summary()

epochs = 30 

callbacks = [
    keras.callbacks.ModelCheckpoint("SavesModel5/save_at_{epoch}.h5"),
    keras.callbacks.EarlyStopping(
    monitor='val_prc', 
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

plot_metrics(history)

#Evaluate metrics before fine tuning.
with open("SavesModel5/Evaluated", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")




#Attempt fine-tuning. Change the base_model to trainable to modify the last few layers.
base_model.trainable = True
for i in model.layers:
    i.trainable = True

#Set learning rate lower since the base model size is much greater than ours.
model.compile(optimizer=keras.optimizers.Adam(1e-5),  
              loss="binary_crossentropy",
              metrics=METRICS)


new_callbacks = [
    keras.callbacks.ModelCheckpoint("SavesModel5/finetuning_save_at_{epoch}.h5", save_weights_only=True),
    keras.callbacks.EarlyStopping(
    verbose=1,
    patience=5,
    restore_best_weights=True)
]
#Train the model again, but allow the base layer to be trained.
history1 = model.fit(train_ds, epochs=15,validation_data=val_ds,callbacks=new_callbacks)


#Refreeze all layers so that we can save it and access the loaded weights from SaveModel5.py. 
base_model.trainable = False
def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

model_freezed = freeze_layers(model)
model_freezed.save('SavesModel5/last_save.h5')

#plot_metrics(history1)

with open("SavesModel5/EvaluatedFineTuning", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")


