#!/usr/bin/env python3
"""
Module to create a Variational Autoencoder (VAE)
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Parameters:
        input_dims (int): dimensions of the model input
        hidden_layers (list): nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: encoder model returning (z, mean, log_variance)
        decoder: decoder model
        auto: full autoencoder model
    """
    # ------------------ ENCODER ------------------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sampling function using the reparameterization trick"""
        mu, log_sig = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu)[0], latent_dims)
        )
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sig])

    encoder = keras.Model(inputs, [z, z_mean, z_log_sig], name='encoder')

    # ------------------ DECODER ------------------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x_dec = latent_inputs

    for nodes in reversed(hidden_layers):
        x_dec = keras.layers.Dense(nodes, activation='relu')(x_dec)

    outputs_dec = keras.layers.Dense(input_dims, activation='sigmoid')(x_dec)

    decoder = keras.Model(latent_inputs, outputs_dec, name='decoder')

    # ------------------ AUTOENCODER ------------------
    encoded_z, mean, log_var = encoder(inputs)
    reconstruction = decoder(encoded_z)

    auto = keras.Model(inputs, reconstruction, name='auto')

    # Add KL divergence loss to autoencoder
    kl_loss = 1 + log_var - keras.backend.square(mean)
    kl_loss -= keras.backend.exp(log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    kl_loss = keras.backend.mean(kl_loss)

    auto.add_loss(kl_loss)

    # Standard compile as strictly requested by task specifications
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
