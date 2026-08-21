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

        Args:
            outputs: list of numpy.ndarrays containing model predictions
            image_size: numpy.ndarray with original image size [height, width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # 1. Box confidence calculation (sigmoid)
            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_conf)

            # 2. Box class probabilities calculation (sigmoid)
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            # 3. Process boundary boxes
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Create grid coordinates matrix
            cx = np.tile(
                np.arange(0, grid_width), (grid_height, 1)
            ).reshape(grid_height, grid_width, 1)

            cy = np.tile(
                np.arange(0, grid_height), (grid_width, 1)
            ).T.reshape(grid_height, grid_width, 1)

            cx = np.tile(cx, (1, 1, anchor_boxes))
            cy = np.tile(cy, (1, 1, anchor_boxes))

            # Apply YOLOv3 transformations relative to grid cell
            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy

            # Normalize center coordinates to model input dimensions
            bx /= grid_width
            by /= grid_height

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bw /= input_width
            bh /= input_height

            # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
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
