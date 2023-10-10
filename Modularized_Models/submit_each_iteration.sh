model="make_model"
output_dir="SavedModel0"
image_size=180
class_weighing="False"
random_rotation=0.2
dropout=0.3
epoch_count=30
transfer_learning="False"
base_model="None"
fine_tuning="False"

python3 "Modularized_Models/main.py" $model $output_dir $image_size $class_weighing $random_rotation $dropout $epoch_count $transfer_learning $base_model $fine_tuning
