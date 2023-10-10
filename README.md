## Pipeline to create a new model:
1. Start with existing code in CVDMachineLearning/Models/classifycvd{some number}.py and copy it to create a new model.
2. Move that model into CVDMachineLearning/
3. Change any parameters required. Make sure to change the output directory to be within a 
SavesModel{new iteration number}/ directory that you make.
4. Run the model with ./run.sh {name_of_model_python_file}
5. If the model is successful, move the model back into CVDMachineLearning/Models/ and the SavesModel directory into SavedModelOutputs/

## Pipeline to Test on Existing Model:
1. Find the path to a saved model, e.g., "CVDMachineLearning/SavedModelOutputs/SavesModel7/first_save.h5"
2. In TestTrainedModels, find the relevant script for running your model on new data. For example, if your model is from iteration 7, use "CVDMachineLearning/TestTrainedModels/saveModel5to7.py". 
3. Change lines 7 through 9 to indicate your test image directory, your model weights, and your output directory.
4. If you wish to run any custom functions to create graphs of the model performance, those can be added at the end of the file. 
5. Execute ./run.sh {name_of_save_model_python_file} at the terminal. For example, "./run.sh TestTrainedModels/saveModel5to7.py


## The General Process for Creating and Refining the Model:
(Check out our paper for a more in-depth explanation of what we did)  
We began this machine learning project after manually annotating and labeling over 5,000 scientific images from the eLife journal. With that data in hand about which images were "Definitely Okay" or "Definitely Problematic" for people with severe deuteranopia, we decided to create a machine learning model to better predict whether future images would be colorblind-friendly or not. We hope that this will serve as a tool for researchers who want their papers to be accessible to a broader audience, including people with color vision deficiencies (CVD).

The first step was to split the 4,360 annotated images (after removing all "Probably Problematic" or "Probably Okay" images from the 5,000 images) into a training dataset and a test dataset (90:10 split). The training dataset has 3,925 images and the test dataset has 435 images. The training dataset was then further split 80:20 into a training and validation set to reduce overfitting while training the convolutional neural network (CNN).

Our initial model (Models/classifycvd0.py) applied data augmentation via random horizontal flipping and image rotation. This model had eight 2D convolutional layers with relu activation functions and a 30% dropout. Every other layer doubled in node size, starting from 32 nodes up to 728. We trained for 30 epochs with an Adam optimization set at an initial learning rate of 1e-3 and the binary cross-entropy loss function. After saving the model and freezing the weights, we evaluated the predictive ability of this initial model using our 435-image testing dataset and used the area under the receiver operating characteristic curve (AUROC) as a metric. The AUROC on the testing set was 0.895.

We iteratively sought to improve the model. Because most images were "Definitely okay" in the training set, the original model was biased toward classifying images as colorblind-friendly. To compensate for this imbalance, we set an initial bias that assigned a higher weight to the minority class (“Definitely problematic” images). As a second adjustment, we added early stopping, which restores the epoch with the highest AUROC; this technique may reduce overfitting. Thirdly, we applied transfer learning by pre-training the model using either ResNet50 or MobileNetV2. MobileNetV2 is a 53-layer convolutional neural network trained on more than a million images from the ImageNet database to classify images with objects into 1,000 categories. ResNet50 is a 50-layer convolutional neural network, similarly trained. MobileNetV2 is designed for use on mobile devices, so it is optimized to be lightweight. As such, MobileNetV2 uses 3.4 million parameters while ResNet50 uses over 25 million trainable parameters.

Of these additions, transfer learning was most effective for improving the model’s performance. It is reasonable that transfer learning would be beneficial because our training dataset contained relatively few images. Using MobileNetV2 returned a slightly higher AUROC (0.913, Models/classifycvd7.py) than ResNet50 (0.903, Models/classifycvd4.py) on the testing dataset. Because of this result and that MobileNetV2 is easier to train, we used MobileNetV2 in all subsequent models. 

We evaluated each model iteration on the test dataset and found that Models/classifycvd7.py performed best with an AUROC of 0.913. This model was built on MobileNetV2 and then we applied a Global Average 2D Pooling layer to the model with a 30% dropout. We then added a dense layer with a sigmoid activation function and trained our final model for 30 epochs with an Adam optimization set at an initial learning rate of 1e-3 and a binary cross-entropy loss function. We then fine-tuned the model by unfreezing the MobileNetV2 neural network layers and retraining the model with a lower initial learning rate of 1e-5. We saved the epoch with the highest AUROC score.

## Navigating the Repository
1. Our model iterations can be found under Models/. Despite setting a random seed as 123, some stochasticity is to be expected (<0.001 change in AUROC) when running the same model due to GPU effects.
2. TestTrainedModels/ include scripts to test model weights on new data.
3. functions.py contains code for plotting model performance and output.
4. findCorruptedFiles.py was used to remove any corrupted images before analysis.
5. run.sh is used to run any Python scripts through docker. It can be executed with './run.sh {name_of_file.py}'
6. BioImageAnalysis/ contains scripts analyzing additional metrics from the images including the proportion of high ratio pixels and Euclidian distance between red and green pixels.
7. ModelAsJSON/ has the saved weights from "Models/classifycvd7.py" exported as a JSON. This folder was used in the website (https://bioapps.byu.edu/colorblind_friendly_tester) to export the model to be used with Javascript.