#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt on a Machine Learning Model
"""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

# CIFAR-10 data preparation for demo/evaluation
(X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
Y_train = K.utils.to_categorical(Y_train, 10)
Y_val = K.utils.to_categorical(Y_val, 10)


def build_and_train_model(lr, units, dropout, l2_reg, batch_size):
    """
    Builds, compiles and trains a simple neural network model
    """
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Flatten()(inputs)
    x = K.layers.Dense(
        int(units),
        activation='relu',
        kernel_regularizer=K.regularizers.l2(l2_reg)
    )(x)
    x = K.layers.Dropout(dropout)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    optimizer = K.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    filename = (f"model_lr{lr:.4f}_units{int(units)}_"
                f"drop{dropout:.2f}_l2{l2_reg:.4f}_bs{int(batch_size)}.h5")

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            filepath=filename,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=int(batch_size),
        epochs=10,
        callbacks=callbacks,
        verbose=0
    )

    best_val_loss = min(history.history['val_loss'])
    return best_val_loss


def objective_function(domain_params):
    """
    Objective function for GPyOpt optimization
    """
    params = domain_params[0]
    lr = float(params[0])
    units = int(params[1])
    dropout = float(params[2])
    l2_reg = float(params[3])
    batch_size = int(params[4])

    val_loss = build_and_train_model(lr, units, dropout, l2_reg, batch_size)
    return val_loss


def main():
    """
    Main execution routine for Bayesian Optimization
    """
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'units', 'type': 'discrete', 'domain': (64, 128, 256)},
        {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=domain,
        acquisition_type='EI',
        exact_feval=True
    )

    max_iter = 30
    optimizer.run_optimization(max_iter=max_iter)

    # Plot convergence and save
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')

    # Save report to bayes_opt.txt
    best_x = optimizer.x_opt
    best_y = optimizer.fx_opt

    report = (
        "Bayesian Optimization Report\n"
        "============================\n"
        f"Best Validation Loss: {best_y}\n"
        f"Optimal Learning Rate: {best_x[0]}\n"
        f"Optimal Units: {int(best_x[1])}\n"
        f"Optimal Dropout: {best_x[2]}\n"
        f"Optimal L2 Reg: {best_x[3]}\n"
        f"Optimal Batch Size: {int(best_x[4])}\n"
    )

    with open('bayes_opt.txt', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
