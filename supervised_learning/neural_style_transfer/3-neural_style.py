#!/usr/bin/env python3
"""
Module defining the NST class for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for Neural Style Transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST

        Args:
            style_image: numpy.ndarray of shape (h, w, 3)
            content_image: numpy.ndarray of shape (h, w, 3)
            alpha: weight for content cost
            beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
           or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) \
           or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels

        Args:
            image: numpy.ndarray of shape (h, w, 3)

        Returns:
            scaled image as tf.Tensor
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(round((w * 512) / h))
        else:
            w_new = 512
            h_new = int(round((h * 512) / w))

        image_expanded = tf.expand_dims(image, axis=0)
        resized_image = tf.image.resize(
            image_expanded,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        rescaled_image = resized_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0.0, 1.0)
        return rescaled_image

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as a
        base. Replaces MaxPooling2D layers with AveragePooling2D
        layers. Saves model to self.model.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        target_layers = self.style_layers + [self.content_layer]

        x = vgg.input
        outputs = []
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
            if layer.name in target_layers:
                outputs.append(x)

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer

        Args:
            input_layer: tf.Tensor or tf.Variable of shape
                (1, h, w, c)

        Returns:
            tf.Tensor of shape (1, c, c) containing the gram
                matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(h * w, tf.float32)
        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets the public instance attributes:
            gram_style_features - a list of gram matrices
                calculated from the style layer outputs of the
                style image
            content_feature - the content layer output of the
                content image
        """
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        style_features = style_outputs[:-1]
        content_feature = content_outputs[-1]

        self.gram_style_features = [
            self.gram_matrix(feature) for feature in style_features
        ]
        self.content_feature = content_feature
