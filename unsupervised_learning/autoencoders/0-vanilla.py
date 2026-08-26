#!/usr/bin/env python3
"""
Module defining a function that creates a Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model

    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing dimensions of latent space

    Returns:
        encoder, decoder, auto:
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_outputs = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=inputs, outputs=latent_outputs)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=outputs)

    # Full Autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
