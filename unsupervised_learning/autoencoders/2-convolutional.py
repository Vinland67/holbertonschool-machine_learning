#!/usr/bin/env python3
"""
Module defining a function that creates a Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model

    Args:
        input_dims: tuple of integers containing dimensions of model input
        filters: list containing the number of filters for each conv layer
                 in the encoder, respectively
        latent_dims: tuple of integers containing dimensions of latent space

    Returns:
        encoder, decoder, auto:
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs=inputs, outputs=x)

    # Decoder
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs

    # Reverse filters except the last one for the main block
    rev_filters = list(reversed(filters))

    # All convolutions except the last two: same padding + upsampling
    for f in rev_filters[:-1]:
        x = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution: valid padding + upsampling
    x = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu', padding='valid'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Last convolution: same number of channels as input_dims, sigmoid
    outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same'
    )(x)

    decoder = keras.Model(inputs=decoder_inputs, outputs=outputs)

    # Full Autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
