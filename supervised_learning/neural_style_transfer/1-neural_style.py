#!/usr/bin/env python3
"""
Module for Neural Style Transfer initialization, image scaling,
and loading the VGG19 model
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST

        Parameters:
        - style_image: numpy.ndarray, style reference image
        - content_image: numpy.ndarray, content reference image
        - alpha: weight for content cost
        - beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or \
           isinstance(alpha, bool) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or \
           isinstance(beta, bool) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels

        Parameters:
        - image: numpy.ndarray of shape (h, w, 3)

        Returns:
        - tf.Tensor scaled image of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray "
                            "with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image_expanded = tf.expand_dims(image, axis=0)

        try:
            resized_image = tf.image.resize_bicubic(
                image_expanded,
                size=[h_new, w_new]
            )
        except AttributeError:
            resized_image = tf.image.resize(
                image_expanded,
                size=[h_new, w_new],
                method=tf.image.ResizeMethod.BICUBIC
            )

        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        Creates the model used to calculate cost.
        Uses VGG19 as a base, replacing MaxPooling2D with AveragePooling2D.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        # Replace MaxPooling2D layers with AveragePooling2D layers
        x = vgg.input
        model_outputs = []
        
        # Reconstruct the model graph to substitute MaxPool with AvgPool
        # and capture target outputs
        outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    name=layer.name
                )
            x = layer(x)
            outputs[layer.name] = x

        target_layers = self.style_layers + [self.content_layer]
        style_and_content_outputs = [outputs[layer] for layer in target_layers]

        model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_and_content_outputs
        )
        self.model = model
