# -*- coding: utf-8 -*-
"""
CVD Machine Learning 6.0
The model is the same as classifiycvd5.py, except we modified our DataAugmentation and Dropout rate.

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


image_size = (224, 224)
batch_size = 32
seed = 123
validation_split=0.20 

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
        layers.RandomRotation(0.2), #Increased randomRotation from 0.1 to 0.2
    ]
)

def make_model(input_shape, output_bias):
    output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) #Increased Dropout rate to 0.3 from 0.2. 

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,) , output_bias=initial_bias)
model.summary()

epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint("Saves/save_at_{epoch}.h5"),
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

plot_metrics(history, "Saves/history")


with open("SavesModel6/Evaluated", "w") as file:
     results = model.evaluate(test_ds)
     for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")

model.save("SavesModel6/first_save.h5")
predictions = model.predict(test_ds)
plot_cm(test_labels, predictions, p=0.5, savePath="SavesModel6/ConfusionMatrix1")

base_model.trainable = True
for i in model.layers:
    i.trainable = True

model.compile(optimizer=keras.optimizers.Adam(1e-5),  
              loss="binary_crossentropy",
              metrics=METRICS)

#Only save last callback from fine-tuned model.

# new_callbacks = [
#     keras.callbacks.ModelCheckpoint("Saves/finetuning_save_at_{epoch}.h5", save_weights_only=True),
#     keras.callbacks.EarlyStopping(
#     verbose=1,
#     patience=5,
#     restore_best_weights=True)
# ]

history1 = model.fit(train_ds, epochs=15,validation_data=val_ds)


base_model.trainable = False

def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

model_freezed = freeze_layers(model)
model_freezed.save('SavesModel6/last_save.h5') #Saves most updated model, includes trained base_model.

plot_metrics(history1, "SavesModel6/history1")


with open("SavesModel6/EvaluatedFineTuning", "w") as file:
    results = model.evaluate(test_ds)
    for name, value in zip(model.metrics_names, results):
        file.write(str(name)+': '+str(value)+"\n")
