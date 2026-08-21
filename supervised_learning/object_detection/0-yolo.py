#!/usr/bin/env python3
"""
Module defining the Yolo class for YOLOv3 object detection
"""
from tensorflow import keras as K


class Yolo:
    """
    Class that uses the YOLO v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class

        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to list of class names used for Darknet model
            class_t: float representing box score threshold
            nms_t: float representing IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
