#!/usr/bin/env python3
"""
Module containing the Yolo class for object detection
"""
import cv2
import glob
import os
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo

        Parameters:
        - model_path: path to Darknet Keras model
        - classes_path: path to list of class names
        - class_t: box score threshold for initial filtering
        - nms_t: IOU threshold for non-max suppression
        - anchors: numpy.ndarray of anchor boxes
        """
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs

        Parameters:
        - outputs: list of numpy.ndarrays containing predictions
        - image_size: numpy.ndarray containing image original size [h, w]

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

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-confidence))
            box_confidences.append(box_confidence)

            classes = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-classes))
            box_class_probs.append(box_class_prob)

            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)

            bx = (1 / (1 + np.exp(-t_x))) + c_x
            by = (1 / (1 + np.exp(-t_y))) + c_y

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            bw = anchor_w * np.exp(t_w)
            bh = anchor_h * np.exp(t_h)

            bx /= grid_width
            by /= grid_height
            bw /= input_width
            bh /= input_height

            x1 = (bx - (bw / 2)) * image_width
            y1 = (by - (bh / 2)) * image_height
            x2 = (bx + (bw / 2)) * image_width
            y2 = (by + (bh / 2)) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes based on box scores threshold

        Parameters:
        - boxes: list of numpy.ndarrays with boundary boxes
        - box_confidences: list of numpy.ndarrays with box confidences
        - box_class_probs: list of numpy.ndarrays with box class probabilities

        Returns:
        tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) on filtered bounding boxes

        Parameters:
        - filtered_boxes: numpy.ndarray of shape (?, 4)
        - box_classes: numpy.ndarray of shape (?,)
        - box_scores: numpy.ndarray of shape (?,)

        Returns:
        tuple of (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        if len(box_predictions) > 0:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )
        else:
            box_predictions = np.array([]).reshape(0, 4)
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder path

        Parameters:
        - folder_path: string representing path to the folder

        Returns:
        tuple of (images, image_paths)
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = [cv2.imread(img_path) for img_path in image_paths]
        return (images, image_paths)
