#!/usr/bin/env python3
"""
Transfer Learning script for CIFAR-10 classification using Keras
"""
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    Parameters:
    - X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 images
    - Y: numpy.ndarray of shape (m, 1) or (m,) containing CIFAR-10 labels

    Returns:
    - X_p: preprocessed X
    - Y_p: preprocessed Y (one-hot encoded)
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def train_model():
    """
    Trains a convolutional neural network using Transfer Learning on CIFAR-10
    and saves the model as cifar10.h5
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Base pretrained model
    base_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    inputs = K.Input(shape=(32, 32, 3))

    # Scale up CIFAR-10 images from 32x32 to 224x224
    scale = K.layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224))
    )(inputs)

    x = base_model(scale, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    # Freeze base model layers initially
    base_model.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        K.callbacks.ModelCheckpoint(
            filepath='cifar10.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    # Warmup top layers
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=128,
        epochs=4,
        callbacks=callbacks
    )

    # Fine-tune base model
    base_model.trainable = True
    # Freeze initial layers of DenseNet121 and unfreeze last layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=128,
        epochs=6,
        callbacks=callbacks
    )


if __name__ == '__main__':
    train_model()
