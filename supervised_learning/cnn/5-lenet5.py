#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras

    X: K.Input of shape (m, 28, 28, 1) containing input images

    Returns: a K.Model compiled to use Adam optimization and accuracy metrics
    """
    initializer = K.initializers.HeNormal(seed=0)

    # 1. Conv1: 6 kernels 5x5, same padding, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # 2. MaxPool1: 2x2 kernel, 2x2 stride
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # 3. Conv2: 16 kernels 5x5, valid padding, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # 4. MaxPool2: 2x2 kernel, 2x2 stride
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(pool2)

    # Flatten
    flatten = K.layers.Flatten()(pool2)

    # 5. FC1: 120 nodes, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # 6. FC2: 84 nodes, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # 7. Output FC3: 10 nodes, Softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Create and compile model
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
