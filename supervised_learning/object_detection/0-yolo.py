#!/usr/bin/env python3
"""
Module containing the Yolo class for object detection
"""
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
