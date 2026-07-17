#!/usr/bin/env python3
"""
Module to build a model using Keras Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using Functional API.
    
    nx: number of input features
    layers: list containing the number of nodes in each layer
    activations: list containing the activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout
    """
    # 1. Giriş təbəqəsini təyin edirik
    inputs = K.Input(shape=(nx,))
    
    regularizer = K.regularizers.L2(lambtha)
    x = inputs
    
    # 2. Təbəqələri dövrlə bir-birinə bağlayırıq
    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)
        
        # Sonuncu təbəqədən başqa hər təbəqədən sonra Dropout tətbiq edirik
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
            
    # 3. Model obyektini yaradırıq (Giriş və çıxışı verməklə)
    model = K.Model(inputs=inputs, outputs=x)
    return model
