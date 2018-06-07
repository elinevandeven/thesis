# show images inline
%matplotlib inline

# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
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

from keras_retinanet.bin.train import create_generators
from keras_retinanet.utils.eval import *
from keras_retinanet.utils.eval import _get_detections
from keras_retinanet.utils.eval import _get_annotations
from keras_retinanet.utils.eval import _compute_ap
from keras_retinanet.preprocessing.csv_generator import *

def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    recalls = {}
    precisions = {}

    #Eline:
    recalls = {}
    precisions = {}
    label_scores = {}
    label_tp = {} 
    label_fp = {}
    label_tp_fn = {}
    label_tp_fp = {}
    
    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        #Eline
        recalls[label] = recall
        precisions[label] = precision
        label_scores[label] = scores
        label_tp[label] = true_positives
        label_fp[label] = false_positives
        label_tp_fn[label] = num_annotations
        label_tp_fp[label] = np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        
        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return (average_precisions, recalls, precisions, label_scores, label_tp, label_fp, label_tp_fn, label_tp_fp)

def compute_f1(label_scores, label_tp, label_fp, label_tp_fn, label_tp_fp):
    diff_0 = np.append(1,np.diff(label_tp[0]))
    diff_1 = np.append(1,np.diff(label_tp[1]))
    diff_2 = np.append(1,np.diff(label_tp[2]))
    diff_all = np.append(np.append(diff_0,diff_1),diff_2)
    ind_0 = np.argsort(-label_scores[0])
    ind_1 = np.argsort(-label_scores[1])
    ind_2 = np.argsort(-label_scores[2])
    all_label_scores = np.append(np.append(label_scores[0][ind_0],label_scores[1][ind_1]),label_scores[2][ind_2])
    indices_2 = np.argsort(-all_label_scores)
    sorted_diff = diff_all[indices_2]
    sorted_sum_diff = list(np.cumsum(sorted_diff))
    
    all_label_scores_2 = np.append(np.append(label_scores[0],label_scores[1]),label_scores[2])
    all_tp = np.append(np.append(label_tp[0],label_tp[1]),label_tp[2])
    all_fp = np.append(np.append(label_fp[0],label_fp[1]),label_fp[2])
    num_annotations = label_tp_fn[0] + label_tp_fn[1] + label_tp_fn[2]

    indices = np.argsort(-all_label_scores_2)
    thresholds = all_label_scores_2[indices]
    sorted_all_tp = sorted_sum_diff
    sorted_all_tp_fn = np.repeat(num_annotations,len(indices))
    sorted_all_tp_fp = np.arange(1.0, len(indices)+1, 1.0)
    
    all_recalls = np.divide(sorted_all_tp, sorted_all_tp_fn)
    all_precisions = np.divide(sorted_all_tp, sorted_all_tp_fp)
    
    teller = np.multiply(all_precisions,all_recalls)
    noemer = np.add(all_precisions,all_recalls)
    f1 = np.multiply(2,np.divide(teller,noemer))
    
    return (f1, thresholds)

## Compute F1 scores for different score confidence thresholds and plot the maximum  
annotation_file = '/home/eline/crops/all_animals/retina_val_crops_all_eline.csv'
class_file = '/home/eline/annotations/mapping_retina_all.csv'
base_dir = '/home/eline/crops/'
generator = CSVGenerator(annotation_file, class_file, base_dir)

average_precisions, recalls, precisions, label_scores, label_tp, label_fp, label_tp_fn, label_tp_fp = evaluate(generator,model)
f1, thresholds = compute_f1(label_scores, label_tp, label_fp, label_tp_fn, label_tp_fp)
max_f1 = f1[np.argmax(f1)]
max_th = thresholds[np.argmax(f1)]
plt.plot(thresholds, f1)
plt.plot(max_th, max_f1,'ko', label = 'Optimal Score Confidence Threshold '+ "{0:0.3f}".format(max_th) + ', F1: ' + "{0:0.3f}".format(max_f1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Score Confidence Threshold')
plt.ylabel('F1-score')
plt.title('F1-score for Different Score Confidence Thresholds')
plt.show()
    
