#!/usr/bin/env python3
"""
Module defining a Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model

    Parameters:
    - input_dims: int, dimensions of the model input
    - hidden_layers: list of int, nodes for each hidden layer in encoder
    - latent_dims: int, dimensions of the latent space representation

    Returns:
    - encoder: encoder model
    - decoder: decoder model
    - auto: full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_output = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs, latent_output)

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
    auto_output = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
