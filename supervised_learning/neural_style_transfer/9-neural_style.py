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
        layers.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        target_layers = self.style_layers + [self.content_layer]

        x = vgg.input
        layer_outputs = {}
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
                layer_outputs[layer.name] = x

        outputs = [layer_outputs[name] for name in target_layers]

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer
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

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c)
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image
        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length)
            )

        weight = 1.0 / length
        J_style = 0
        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):
            J_style += weight * self.layer_style_cost(
                style_output, gram_target)

        return J_style

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != s:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != s:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != s:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                        beta1=0.9, beta2=0.99):
        """
        Generates the neural style transferred image

        iterations: number of iterations to perform gradient
            descent over
        step: if not None, the step at which to print training
            information
        lr: learning rate for gradient descent
        beta1: beta1 parameter for gradient descent
        beta2: beta2 parameter for gradient descent

        Returns: generated_image, cost
            generated_image: the best generated image
            cost: the best cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style = self.compute_grads(
                generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])
            clipped = tf.clip_by_value(generated_image, 0.0, 1.0)
            generated_image.assign(clipped)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, "
                      "style {}".format(i, J_total, J_content, J_style))

        best_image = best_image[0]
        return best_image, best_cost
