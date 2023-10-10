from models import * 

##Run each model

model0 = {
    'model_function':make_model,
    'output_folder':"SavedModel0",
    'image_size':180,
    'include_class_weighing':False,
    'early_stopping':False,
    'random_rotation':0.2,
    'dropout':0.3,
    'epoch_count':30,
    'transfer_learning':False,
    'transfer_learning_model':None,
    'fine_tuning':False
}

model1=model0.copy()
model1['include_class_weighing'] = True
model1['output_folder'] = "SavedModel1"


model2 = model1.copy()
model2['early_stopping'] = True
model2['output_folder'] = "SavedModel2"


model3 = model2.copy()
model3['random_rotation'] = 0.1
model3['output_folder'] = "SavedModel3"


model4 = model3.copy()
model4['random_rotation'] = 0.2
model4["model_function"] = make_model_resnet_50
model4['transfer_learning'] =  True
model4['transfer_learning_model'] = "ResNet50"
model4['output_folder'] = "SavedModel4"


model4_1 = model4.copy()
model4_1["image_size"] = 224
model4_1["fine_tuning"] = True
model4_1['output_folder'] = "SavedModel4_1"

model5 = model4_1.copy()
model5["model_function"] = make_model_mobile_net2
model5['output_folder'] = "SavedModel5"


model6 = model5.copy()
model6["dropout"] = 0.2
model6['output_folder'] = "SavedModel6"


#Sanity check
model7 = model6.copy()
model7['output_folder'] = "SavedModel7"

model10 = model7.copy()
model10["image_size"] = 512
model10['output_folder'] = "SavedModel10"


#for num, model in enumerate([model0,model4,model4_1,model5,model6,model7,model10]):
for num, model in enumerate([model0, model1,model2,model3,model4,model4_1,model5,model6,model7,model10]):
    try:
        run_model(**model)
    except Exception as e:
        print(e)
        with open("error.log", "w") as logFile:
            logFile.write(str(num))
            logFile.write(str(model))
            logFile.write(e)
    