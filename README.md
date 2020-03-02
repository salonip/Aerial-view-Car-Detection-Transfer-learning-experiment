# Aerial-view-Car-Detection-Transfer-learning-experiment
- The dataset downloaded here is COWC - only used POSTDAM data for simplicity
- python create_dataset.py takes image folder as input and ouputs train.csv and test.csv
- This will annotate the images using _annotated_cars images in cowc dataset
- # Create train data:
  python get_tfrecord.py --csv_input=Cowc_car_counting/data/train_labels.csv  --output_path=Cowc_car_counting/data/train.record --image_dir=/home/navoni/Cowc_car_counting/data/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/

  # Create test data:
  python get_tfrecord.py --csv_input=Cowc_car_counting/data/test_labels.csv  --output_path=Cowc_car_counting/data/test.record --image_dir=/home/navoni/Cowc_car_counting/data/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/
- Download pretrained model and update pipeline.config
# Set up tensorflow/models/ and execute commands 
PIPELINE_CONFIG_PATH=training/pipeline.config
MODEL_DIR=Cowc_car_counting/models/
NUM_TRAIN_STEPS=500
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
cd tensorflow/models/research/
python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH}  --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS}  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr

There you go!

Model trained. Now convert them into .pb files. For that, execute,

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=pipeline.config
TRAINED_CKPT_PREFIX=model.ckpt-305
EXPORT_DIR=export/
python3 object_detection/export_inference_graph.py --input_type=${INPUT_TYPE}   --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}


Now model is ready!

use the following for using model on images
python train_and_predict.py
Done!
