#!/usr/bin/env python3
"""
Module defining the NST class for Neural Style Transfer
"""
import tensorflow as tf
import numpy as np


class NST:
    """
    Class that performs tasks for Neural Style Transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer

        Args:
            style_image: image used as a style reference
            content_image: image used as a content reference
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

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
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
        Rescales an image such that its pixels are in range [0, 1] and its
        largest side is 512 pixels.

        Args:
            image: numpy.ndarray of shape (h, w, 3)

        Returns:
            scaled_image: tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int((w * 512) / h)
        else:
            w_new = 512
            h_new = int((h * 512) / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """
        Creates the model used for Neural Style Transfer
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        custom_objects = {
            'AveragePooling2D': tf.keras.layers.AveragePooling2D
        }

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        self.model = model

    def generate_features(self):
        """
        Extracts the features used to calculate neural style transfer cost
        """
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_preprocessed)
        content_outputs = self.model(content_preprocessed)

        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates the Gram matrix of a tensor

        Args:
            input_tensor: tf.Tensor of shape (1, h, w, c) or (h, w, c)

        Returns:
            gram: tf.Tensor of shape (1, c, c)
        """
        if not isinstance(input_tensor, (tf.Tensor, np.ndarray)) or \
           input_tensor.ndim not in [3, 4]:
            raise TypeError("input_tensor must be a tensor of rank 3 or 4")

        if input_tensor.ndim == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)

        num_locations = tf.cast(
            tf.shape(a)[0],
            dtype=tf.float32
        )
        return gram / num_locations

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        Args:
            style_output: tf.Tensor of shape (1, h, w, c)
            gram_target: tf.Tensor of shape (1, c, c)

        Returns:
            cost: tf.Tensor containing layer style cost
        """
        if not isinstance(style_output, (tf.Tensor, np.ndarray)) or \
           style_output.ndim != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]

        if not isinstance(gram_target, (tf.Tensor, np.ndarray)) or \
           gram_target.ndim != 3 or \
           gram_target.shape[1] != c or \
           gram_target.shape[2] != c:
            raise TypeError(
                f"gram_target must be a tensor of shape (1, {c}, {c})"
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the total style cost for all style layers

        Args:
            style_outputs: list of tf.Tensor containing style outputs

        Returns:
            J_style: total style cost
        """
        num_style_layers = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != num_style_layers:
            raise TypeError(
                f"style_outputs must be a list of length {num_style_layers}"
            )

        style_weights = [1.0 / num_style_layers] * num_style_layers
        J_style = 0.0

        for style_output, gram_target, weight in zip(
            style_outputs, self.gram_style_features, style_weights
        ):
            layer_cost = self.layer_style_cost(style_output, gram_target)
            J_style += weight * layer_cost

        return J_style

    def content_cost(self, content_output):
        """
        Calculates the content cost for the content layer

        Args:
            content_output: tf.Tensor containing content output

        Returns:
            J_content: content cost
        """
        if not isinstance(content_output, (tf.Tensor, np.ndarray)) or \
           content_output.ndim != 4 or \
           content_output.shape[1:] != self.content_feature.shape[1:]:
            raise TypeError(
                "content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, h, w, 3)

        Returns:
            J_total, J_content, J_style
        """
        s_g = generated_image.shape
        if not isinstance(generated_image, (tf.Tensor, np.ndarray)) or \
           generated_image.ndim != 4 or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        gen_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        outputs = self.model(gen_preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        J_total = self.alpha * J_content + self.beta * J_style

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, h, w, 3)

        Returns:
            gradients, J_total, J_content, J_style
        """
        if not isinstance(generated_image, (tf.Tensor, np.ndarray)) or \
           generated_image.ndim != 4 or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
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

        Args:
            iterations: number of iterations to perform gradient descent over
            step: step at which to print information about training
            lr: learning rate for Adam optimizer
            beta1: beta1 parameter for Adam optimizer
            beta2: beta2 parameter for Adam optimizer

        Returns:
            generated_image, cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0.0 <= beta1 <= 1.0):
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0.0 <= beta2 <= 1.0):
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image
            )
            optimizer.apply_gradients([(grads, generated_image)])

            clipped = tf.clip_by_value(generated_image, 0.0, 1.0)
            generated_image.assign(clipped)

            if J_total < best_cost:
                best_cost = J_total.numpy()
                best_image = generated_image.numpy()

            if step is not None and (i == 0 or i % step == 0 or
                                     i == iterations):
                print(
                    f"Cost at iteration {i}: {J_total},"
                    f" content {J_content}, style {J_style}"
                )

        # best_image matrisindən 1-ci dimensiyanı (1, h, w, 3) -> (h, w, 3) çıxarırıq
        best_image = best_image[0]

        return best_image, best_cost
