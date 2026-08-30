#!/usr/bin/env python3
"""
Module containing the Yolo class for object detection
"""
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
