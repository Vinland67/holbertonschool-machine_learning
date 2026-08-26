#!/usr/bin/env python3
"""
Module defining a function that creates a Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model

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

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """
        Sampling layer for reparameterization trick
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

    def vae_loss(x, x_decoded):
        """
        Calculates custom VAE loss combining binary cross-entropy
        reconstruction loss and KL divergence
        """
        reconstruction_loss = keras.losses.binary_crossentropy(x, x_decoded)
        reconstruction_loss *= input_dims

        kl_loss = 1 + log_sig - keras.backend.square(mean) - \
            keras.backend.exp(log_sig)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
