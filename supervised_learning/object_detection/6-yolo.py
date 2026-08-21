#!/usr/bin/env python3
"""
Module defining the Yolo class for YOLOv3 object detection
"""
import cv2
import glob
import numpy as np
import os
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
        """
        filtered_boxes = None
        box_classes = None
        box_scores = None

        for box, box_conf, box_prob in zip(boxes, box_confidences,
                                           box_class_probs):
            scores = box_conf * box_prob
            b_classes = np.argmax(scores, axis=-1)
            b_scores = np.max(scores, axis=-1)

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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to filtered bounding boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for c in unique_classes:
            idx = np.where(box_classes == c)
            c_boxes = filtered_boxes[idx]
            c_scores = box_scores[idx]

            x1 = c_boxes[:, 0]
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            area = (x2 - x1) * (y2 - y1)

            order = c_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                if order.size == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (area[i] + area[order[1:]] - inter)

                inds = np.where(iou < self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(c_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), c))
            predicted_box_scores.append(c_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder path
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for Darknet model input
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append((img.shape[0], img.shape[1]))
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names,
        and box scores.

        Args:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing boundary boxes for image
            box_classes: numpy.ndarray containing class indices for boxes
            box_scores: numpy.ndarray containing box scores for boxes
            file_name: file path where original image is stored
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]

            # Draw blue bounding box (thickness 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Text format: class_name score (score rounded to 2 decimals)
            text = f"{class_name} {score:.2f}"
            org = (x1, y1 - 5)

            # Draw red text above top left corner
            cv2.putText(
                image, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        # Check for 's' key press (ASCII 115 / ord('s'))
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()
