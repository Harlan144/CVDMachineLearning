import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import glob
import os
import matplotlib.pyplot as plt
#from functions import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50


from keras.models import Model
from numpy.random import seed
import sys

## All Model functions

def make_model_mobile_net2(input_shape, output_bias, data_augmentation,base_model, dropout=0.3):
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias) #Use the initial_bias as defined above. Decreases initial loss.
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False) #Call our MobileNetV2 base_model.

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x) #Used to decrease overfitting

    outputs = layers.Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)

def make_model_resnet_50(input_shape, output_bias, data_augmentation, base_model, dropout=0.3):

    output_bias = tf.keras.initializers.Constant(output_bias)
    
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) #apply augmentation

    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x) #Lowered dropout rate from 0.5 to 0.3.

    outputs = layers.Dense(1, activation="sigmoid",bias_initializer=output_bias)(x)
    return keras.Model(inputs, outputs)


def make_model(input_shape, output_bias, data_augmentation, base_model, dropout=0.3):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

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
    x = layers.Dropout(dropout)(x) #Reduce overfitting

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

def run_model(model_function = make_model_mobile_net2,
              output_folder = "SavedModel", 
              image_size=224,
              include_class_weighing=False, 
              early_stopping=False,
              random_rotation = 0.2, 
              dropout=0.3, 
              epoch_count=30,
              transfer_learning=False,
              transfer_learning_model ="",
              fine_tuning = False
              ):

    image_size = (image_size, image_size) #Changed from (224,224)
    batch_size = 32
    seed = 123 #Set our random seed as 123 to ensure reproducibility
    validation_split=0.20 #Always use 0.0


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

    #Take 20% of train_ds and withhold as validation through the testing.
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

    initial_bias = None #This will be set to 1 or the class_weight

    if include_class_weighing:
        #Apply a weight 
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
    else:
        initial_bias=1

    

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=123),
            layers.RandomRotation(random_rotation, seed = 123),
        ]
    )

    base_model = None

    if transfer_learning:
        if transfer_learning_model =="MobileNetV2":
        
            base_model = MobileNetV2(input_shape=image_size+(3,),
            include_top=False,
            weights='imagenet')

            base_model.trainable = False

        elif transfer_learning_model =="ResNet50":
            base_model = ResNet50(weights='imagenet', 
                             input_shape= image_size+(3,),
                             include_top=False) 
            
            base_model.trainable = False
    

    model = model_function(input_shape=image_size+ (3,), output_bias=initial_bias, data_augmentation=data_augmentation, base_model=base_model, dropout=dropout)
    
    model.summary() #Print a summary of the model onto the command line.

    if early_stopping:
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(output_folder,"save_at_{epoch}.h5")),
            keras.callbacks.EarlyStopping(
            monitor='val_auc', #Stops when validation AUC is the highest.
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)
        ]
    else:
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(output_folder,"save_at_{epoch}.h5")),
        ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=METRICS,
    )


#Call the model. Use defined class weights to improve accuracy.
    if include_class_weighing:
        history = model.fit(
            train_ds,
            epochs=epoch_count,
            callbacks=callbacks,
            validation_data=val_ds,
            class_weight=class_weight
        )
    else:
        history = model.fit(
            train_ds,
            epochs=epoch_count,
            callbacks=callbacks,
            validation_data=val_ds,
        )
    

    print(history.history.keys())

    historyDf = pd.DataFrame.from_dict(history.history)
    historyDf.to_csv(os.path.join(output_folder,"History.tsv"), sep="\t")

    with open(os.path.join(output_folder,"Evaluated"), "w") as file:
        results = model.evaluate(test_ds)
        for name, value in zip(model.metrics_names, results):
            file.write(str(name)+': '+str(value)+"\n")
    

    model.save(os.path.join(output_folder,"final_save.h5"))


    predictions = model.predict(test_ds)

    with open(os.path.join(output_folder,"Predictions"),"w") as output:
        for i in range(len(predictions)):
            if test_labels[i]==1:
                output.write(f"Unfriendly\t{float(predictions[i])}\n")
            else:
                output.write(f"Friendly\t{float(predictions[i])}\n")

    if fine_tuning and transfer_learning:
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

        history1 = model.fit(train_ds, epochs=2,validation_data=val_ds, callbacks= new_callbacks)
        print(history1.history.keys())

        historyDf1 = pd.DataFrame.from_dict(history1.history)
        historyDf1.to_csv(os.path.join(output_folder,"History_FineTuning.tsv"), sep="\t")
        
        #Freeze all layers of our model so that we can save it.
        base_model.trainable = False
        
        model_freezed = freeze_layers(model)

        model_freezed.save(os.path.join(output_folder,"finetuning_save.h5"))

        with open(os.path.join(output_folder,"Evaluated_Finetuning"), "w") as file:
            results = model_freezed.evaluate(test_ds)
            for name, value in zip(model_freezed.metrics_names, results):
                file.write(str(name)+': '+str(value)+"\n")

        predictions_finetuned = model_freezed.predict(test_ds)

        with open(os.path.join(output_folder,"Predictions_Finetuning"),"w") as output:
            for i in range(len(predictions_finetuned)):
                if test_labels[i]==1:
                    output.write(f"Unfriendly\t{float(predictions_finetuned[i])}\n")
                else:
                    output.write(f"Friendly\t{float(predictions_finetuned[i])}\n")

    else:
        print("No finetuning performed.")
    




    
seed(123) #Set random seed with numpy
tf.random.set_seed(123) #Set random seed with tensorflow 

# run_model(
#     epoch_count = 2,
#     transfer_learning=True,
#     transfer_learning_model ="MobileNetV2",
#     fine_tuning = True
# )
