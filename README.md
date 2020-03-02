# Aerial-view-Car-Detection-Transfer-learning-experiment
- The dataset downloaded here is COWC - only used POSTDAM data for simplicity
    - wget ftp://gdo152.ucllnl.org/pub/cowc/Postdam will give you the data.
    - Alternatively, you can look here for scripts and Licence https://github.com/LLNL/cowc/blob/master/README.md 
- Setup for Tensorflow/models
    - $ pip install tensorflow
    - $ pip install pillow Cython lxml jupyter matplotlib
    - cd <path_to_your_tensorflow_installation> (usr/local/lib/python3/dist_packages/tensorflow for me)
    - git clone https://github.com/tensorflow/models.git
    - $ cd <path_to_your_tensorflow_installation>/models/research/
    - $ protoc object_detection/protos/*.proto --python_out=.
    - $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    - Verify installation by $ python object_detection/builders/model_builder_test.py

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
