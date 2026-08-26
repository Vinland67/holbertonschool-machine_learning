#!/usr/bin/env python3
"""
Module defining a function that creates a Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """
        Sampling layer using reparameterization trick
        """
        z_mean, z_log_sig = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(z_log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_sig])
    encoder = keras.Model(inputs=inputs, outputs=[z, mean, log_sig])

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=outputs)

    # Full Autoencoder
    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    # Loss Definition
    reconstruction_loss = keras.backend.binary_crossentropy(
        inputs, auto_outputs
    )
    reconstruction_loss = keras.backend.sum(
        reconstruction_loss, axis=-1
    )

    kl_loss = 1 + log_sig - keras.backend.square(mean) - \
        keras.backend.exp(log_sig)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
