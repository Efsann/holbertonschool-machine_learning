#!/usr/bin/env python3
"""
Module defining a Public Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a public variational autoencoder model

    Parameters:
    - input_dims: int, dimensions of the model input
    - hidden_layers: list of int, nodes for each hidden layer in encoder
    - latent_dims: int, dimensions of the latent space representation

    Returns:
    - encoder: encoder model outputting latent representation, mean, log_var
    - decoder: decoder model
    - auto: full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sampling function for reparameterization trick"""
        mu, log_sig = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.shape(mu)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x_dec = latent_inputs
    for nodes in reversed(hidden_layers):
        x_dec = keras.layers.Dense(nodes, activation='relu')(x_dec)
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(x_dec)
    decoder = keras.Model(latent_inputs, decoded_output)

    # Full Autoencoder
    auto_output = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_output)

    def vae_loss(x_in, x_out):
        """VAE Custom Loss Function (Reconstruction + KL divergence)"""
        recon_loss = keras.losses.binary_crossentropy(x_in, x_out)
        recon_loss *= input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_sigma),
            axis=-1
        )
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer=keras.optimizers.Adam(), loss=vae_loss)

    return encoder, decoder, auto
