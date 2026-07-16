#!/usr/bin/env python3
"""
Module to build a sequential neural network using Keras
"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    model = tf.keras.Sequential()
    
    # İlk qatı input_dim (nx) ilə əlavə edirik
    model.add(tf.keras.layers.Dense(
        layers[0],
        input_dim=nx,
        activation=activations[0],
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    ))
    
    # Əgər keep_prob < 1-dirsə, Dropout əlavə edirik
    if keep_prob < 1:
        model.add(tf.keras.layers.Dropout(1 - keep_prob))
        
    # Digər qatları ardıcıl olaraq əlavə edirik
    for i in range(1, len(layers)):
        model.add(tf.keras.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(lambtha)
        ))
        if keep_prob < 1 and i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1 - keep_prob))
            
    return model
