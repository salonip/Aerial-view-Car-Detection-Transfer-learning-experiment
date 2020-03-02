'''

 sudo cp /home/navoni/Downloads/ssd_mobilenet_v3_small_coco_2019_08_14/*.ckpt.* /usr/local/lib/python3.6/dist-packages/tensorflow/models/checkpoints/
PIPELINE_CONFIG_PATH=/home/navoni/Downloads/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config
MODEL_DIR=/home/navoni/Cowc_car_counting/models/
NUM_TRAIN_STEPS=5000
TF_CPP_MIN_LOG_LEVEL=2
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

sudo python3 object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH}  --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS}  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr


export PYTHONPATH="$PYTHONPATH:/home/navoni/tensorflow/models/research:/home/navoni/tensorflow/models/research/slim

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/navoni/Downloads/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config
TRAINED_CKPT_PREFIX=/home/navoni/Cowc_car_counting/models/model.ckpt-1160
EXPORT_DIR=/home/navoni/Cowc_car_counting/export/
sudo python3 object_detection/export_inference_graph.py --input_type=${INPUT_TYPE}   --pipeline_config_path=${PIPELINE_CONFIG_PATH} --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} --output_directory=${EXPORT_DIR}



'''

import numpy as np
import tensorflow as tf
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

parser = argparse.ArgumentParser(description='Run object recognition')
parser.add_argument('--modelpath', type=str,default='../export_1276/',
                    help='directory for frozen interference graph')
parser.add_argument('--imagepath', default='../data/ml_test.jpg',
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

# Read the graph.
with tf.gfile.FastGFile(args.modelpath+'frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.

    fname = str(args.imagepath).split('.png')[0]
    img = cv.imread(args.imagepath)

    #img = cv.resize(img, (2220,2220))
    inp = cv.resize(img, (1024,600))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    #img = cv.resize(img, (1024,600))
    rows = img.shape[0]
    cols = img.shape[1]
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    det=0
    img = cv.resize(img, (cols,rows))
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score >= 0.35:
            det = det+1
            x = bbox[1] * cols
            y = bbox[0] * rows
            #right = x+60
            #bottom = y+60
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            
            cv.putText(img,str(score)[0:5], (int(x)+5, int(y)+5), cv.FONT_HERSHEY_SIMPLEX ,1, (0, 0, 255) , 2, cv.LINE_AA) 
            #cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv.circle(img, (int(x)+int((right-x)/2), int(y)+int((bottom-y)/2)), 7,(0, 0, 255), thickness=2)

print('Cars detected : '+str(det))
cv.imwrite(fname+'_annot.png', img)
