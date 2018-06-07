# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

from PIL import ImageDraw
from PIL import Image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load retinanet model
model_name = 'resnet50_csv_03.h5'
model_path = os.path.join('/home/eline/snapshots/all_animals_crops/',model_name)
model = keras.models.load_model(model_path, custom_objects=custom_objects)
labels_to_names = {0: 'zebra', 1: 'giraffe', 2:'elephant'}

#load annotations
annotations_file =  '/home/eline/crops/all_animals/retina_test_crops_all_eline.csv'

annotations = []
crops = []
with open(annotations_file, 'r') as f:
    for line in f:
        annotation = line.split(',')
        annotations.append(annotation)
        crop_path = annotation[0]
        crops.append(crop_path)
crops = list(set(crops))

# save all annotations per crop in one object
crop_gt_boxes = {}
crop_gt_labels = {}
crop_det_boxes = {}
crop_det_labels = {}

for crop in crops:
    gt_boxes = []
    gt_labels = []
    with open(annotations_file, 'r') as f:
        for line in f:
            annotation = line.split(',')
            crop_i = annotation[0]
            if crop_i == crop:
                gt_boxes.append(map(int,annotation[1:5]))
                gt_labels.append(annotation[5])
    
    crop_gt_boxes[crop] = gt_boxes
    crop_gt_labels[crop] = gt_labels
    
    image_path = crop
    image = read_image_bgr(image_path)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
   
    # process image
    start = time.time()
    _, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))

    print(crop, " processing time: ", time.time() - start)
    
    # compute predicted labels and scores
    predicted_labels = np.argmax(nms_classification[0, :, :], axis=1)
    scores = nms_classification[0, np.arange(nms_classification.shape[1]), predicted_labels]
    
    # correct for image scale
    boxes /= scale
              
    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.571: #0.5:
            if crop not in crop_det_boxes:
                crop_det_boxes[crop] = []
                crop_det_labels[crop] = []
            continue            
        color = label_color(label)
        b = boxes[0, idx, :].astype(int)
        if not crop in crop_det_boxes:
            crop_det_boxes[crop] = [b]
            crop_det_labels[crop] = [labels_to_names[label]]
        else:
            crop_det_boxes[crop].append(b)
            crop_det_labels[crop].append(labels_to_names[label])
            
det_gt = []
missed_detections = []
for crop, detections in crop_det_boxes.iteritems():
    detections = crop_det_boxes[crop]
    gt_boxes = crop_gt_boxes[crop]
    detected_annotations = []
    for ix, d in enumerate(detections):
        annotations = np.array(crop_gt_boxes[crop])             
    
        if annotations.shape[0] == 0:
            det_get.append([crop_det_labels[crop][ix], 'background']) 
            continue

        overlaps = compute_overlap(np.expand_dims(d, axis=0),np.array(gt_boxes))
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]
        if max_overlap >= 0.5 and gt_boxes[assigned_annotation[0]] not in detected_annotations:
            det_gt.append([crop_det_labels[crop][ix],crop_gt_labels[crop][assigned_annotation[0]]])
            detected_annotations.append(gt_boxes[assigned_annotation[0]])
        else:
            det_gt.append([crop_det_labels[crop][ix],'background'])
    
    for i, gb in enumerate(gt_boxes):
        if gb not in detected_annotations:
            det_gt.append(['background',crop_gt_labels[crop][i]])
            missed_detections.append(gb)

for x in set(map(tuple, det_gt)):
    print '{} = {}'.format(x, det_gt.count(list(x))) 
