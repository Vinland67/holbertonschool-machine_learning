#!/usr/bin/env python3
"""
Module defining the Yolo class for YOLOv3 object detection
"""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """
    Class that uses the YOLO v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs to extract bounding boxes,
        box confidences, and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_conf)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            cx = np.tile(
                np.arange(0, grid_width), (grid_height, 1)
            ).reshape(grid_height, grid_width, 1)

            cy = np.tile(
                np.arange(0, grid_height), (grid_width, 1)
            ).T.reshape(grid_height, grid_width, 1)

            cx = np.tile(cx, (1, 1, anchor_boxes))
            cy = np.tile(cy, (1, 1, anchor_boxes))

            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy

            bx /= grid_width
            by /= grid_height

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bw /= input_width
            bh /= input_height

            x1 = (bx - (bw / 2)) * image_width
            y1 = (by - (bh / 2)) * image_height
            x2 = (bx + (bw / 2)) * image_width
            y2 = (by + (bh / 2)) * image_height

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters bounding boxes based on objectness score threshold

        Args:
            boxes: list of numpy.ndarrays of shape
                   (grid_height, grid_width, anchor_boxes, 4)
            box_confidences: list of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: list of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, classes)

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = None
        box_classes = None
        box_scores = None

        for box, box_conf, box_prob in zip(boxes, box_confidences,
                                           box_class_probs):
            # Calculate box scores: confidence * class probabilities
            scores = box_conf * box_prob

            # Find predicted class and highest class score for each box
            b_classes = np.argmax(scores, axis=-1)
            b_scores = np.max(scores, axis=-1)

            # Mask boxes that satisfy threshold
            mask = b_scores >= self.class_t

            f_boxes = box[mask]
            f_classes = b_classes[mask]
            f_scores = b_scores[mask]

            if filtered_boxes is None:
                filtered_boxes = f_boxes
                box_classes = f_classes
                box_scores = f_scores
            else:
                filtered_boxes = np.concatenate((filtered_boxes, f_boxes),
                                                axis=0)
                box_classes = np.concatenate((box_classes, f_classes),
                                             axis=0)
                box_scores = np.concatenate((box_scores, f_scores),
                                            axis=0)

        return filtered_boxes, box_classes, box_scores
