#!/usr/bin/env python3
"""
Module defining a Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model

    Parameters:
    - input_dims: tuple of int, dimensions of the model input
    - filters: list of int, number of filters for each conv layer in encoder
    - latent_dims: tuple of int, dimensions of latent space representation

    Returns:
    - encoder: encoder model
    - decoder: decoder model
    - auto: full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)
    encoder = keras.Model(inputs, x)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x_dec = latent_inputs
    rev_filters = list(reversed(filters))

    # All decoder conv layers except the last two
    for f in rev_filters[:-1]:
        x_dec = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x_dec)
        x_dec = keras.layers.UpSampling2D(size=(2, 2))(x_dec)

    # Second to last convolution (valid padding + upsampling)
    x_dec = keras.layers.Conv2D(
        filters=rev_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x_dec)
    x_dec = keras.layers.UpSampling2D(size=(2, 2))(x_dec)

    # Last convolution (same padding, sigmoid activation, no upsampling)
    decoded_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x_dec)

    decoder = keras.Model(latent_inputs, decoded_output)

    # Full Autoencoder
    auto_output = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
