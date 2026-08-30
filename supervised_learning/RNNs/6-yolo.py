#!/usr/bin/env python3
"""
Module defining the Yolo class for object detection
"""
import os
import cv2
import numpy as np
import tensorflow.keras as keras


class Yolo:
    """
    Yolo v3 class to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo class
        """
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes darknet outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.tile(
                np.arange(grid_width).reshape(1, grid_width, 1),
                (grid_height, 1, 1)
            )
            c_y = np.tile(
                np.arange(grid_height).reshape(grid_height, 1, 1),
                (1, grid_width, 1)
            )

            c_x = np.tile(c_x[..., np.newaxis], (1, 1, anchor_boxes))
            c_y = np.tile(c_y[..., np.newaxis], (1, 1, anchor_boxes))

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            input_width = self.model.input.shape[1].value
            if input_width is None:
                input_width = self.model.input.shape[1]

            input_height = self.model.input.shape[2].value
            if input_height is None:
                input_height = self.model.input.shape[2]

            b_w = (anchor_w * np.exp(t_w)) / input_width
            b_h = (anchor_h * np.exp(t_h)) / input_height

            x1 = (b_x - (b_w / 2)) * image_width
            y1 = (b_y - (b_h / 2)) * image_height
            x2 = (b_x + (b_w / 2)) * image_width
            y2 = (b_y + (b_h / 2)) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters output boundary boxes based on confidence threshold
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def _iou(self, box1, box2):
        """
        Calculates Intersection over Union (IoU) between two bounding boxes
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0

        return intersection / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to predicted bounding boxes
        """
        keep_boxes = []
        keep_classes = []
        keep_scores = []

        unique_classes = np.unique(box_classes)

        for c in unique_classes:
            class_mask = box_classes == c

            c_boxes = filtered_boxes[class_mask]
            c_classes = box_classes[class_mask]
            c_scores = box_scores[class_mask]

            order = np.argsort(c_scores)[::-1]

            c_boxes = c_boxes[order]
            c_classes = c_classes[order]
            c_scores = c_scores[order]

            while len(c_boxes) > 0:
                keep_boxes.append(c_boxes[0])
                keep_classes.append(c_classes[0])
                keep_scores.append(c_scores[0])

                if len(c_boxes) == 1:
                    break

                ious = np.array(
                    [self._iou(c_boxes[0], box) for box in c_boxes[1:]]
                )

                iou_mask = ious < self.nms_t
                c_boxes = c_boxes[1:][iou_mask]
                c_classes = c_classes[1:][iou_mask]
                c_scores = c_scores[1:][iou_mask]

        return (
            np.array(keep_boxes),
            np.array(keep_classes),
            np.array(keep_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a given folder
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for Darknet model input
        """
        input_width = self.model.input.shape[1].value
        if input_width is None:
            input_width = self.model.input.shape[1]

        input_height = self.model.input.shape[2].value
        if input_height is None:
            input_height = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            h, w, _ = img.shape
            image_shapes.append((h, w))

            resized_img = cv2.resize(
                img,
                (input_width, input_height),
                interpolation=cv2.INTER_CUBIC
            )

            rescaled_img = resized_img / 255.0
            pimages.append(rescaled_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores

        Parameters:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing the boundary boxes
            box_classes: numpy.ndarray containing class indices for each box
            box_scores: numpy.ndarray containing box scores for each box
            file_name: file path where the original image is stored
        """
        # Copy image to avoid modifying original array
        img_copy = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]

            # Draw box in blue (BGR: (255, 0, 0)) with thickness 2
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Format string with class name and score rounded to 2 decimal places
            label = f"{class_name} {score:.2f}"

            # Text positioned 5 pixels above top-left corner
            text_pos = (x1, y1 - 5)

            # Draw text in red (BGR: (0, 0, 255))
            cv2.putText(
                img_copy,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # Display window
        cv2.imshow(file_name, img_copy)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        # If 's' key is pressed, save image in detections folder
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, img_copy)

        # Close image window
        cv2.destroyAllWindows()
