#!/usr/bin/env python3
"""
Mövzu: Keras ilə Sequential neyron şəbəkəsinin qurulması
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Keras kitabxanası istifadə edərək neyron şəbəkəsi qurur.
    
    nx: şəbəkənin giriş xüsusiyyətlərinin sayı
    layers: hər təbəqədəki düyünlərin sayı (siyahı)
    activations: hər təbəqənin aktivasiya funksiyası (siyahı)
    lambtha: L2 requlyarizasiya parametri
    keep_prob: Dropout üçün düyünün saxlanılma ehtimalı
    """
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)
    
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))
        
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
            
    return model

