#!/usr/bin/env python3
"""
Bayesian Optimization of a Deep Learning Model using GPyOpt
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import GPyOpt
import matplotlib.pyplot as plt


def create_and_train_model(x):
    """
    Objective function for GPyOpt
    x is a 2D numpy array containing hyperparameter values:
    x[:, 0] -> learning_rate
    x[:, 1] -> num_units
    x[:, 2] -> dropout_rate
    x[:, 3] -> l2_reg
    x[:, 4] -> batch_size
    """
    # Extract hyperparameters
    lr = float(x[:, 0][0])
    num_units = int(x[:, 1][0])
    dropout_rate = float(x[:, 2][0])
    l2_reg = float(x[:, 3][0])
    batch_size = int(x[:, 4][0])

    # Load MNIST dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_val = x_val.reshape(-1, 28 * 28)

    # Build model
    model = models.Sequential([
        layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=(784,)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(
            num_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Checkpoint filename specifying hyperparameter values
    checkpoint_filename = (
        f"model_lr{lr:.4f}_units{num_units}_drop{dropout_rate:.2f}_"
        f"l2{l2_reg:.4f}_bs{batch_size}.h5"
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filename,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=0
    )

    # Satisficing metric: Validation Error (1 - Validation Accuracy)
    best_val_acc = max(history.history['val_accuracy'])
    val_error = 1.0 - best_val_acc

    return val_error


def main():
    """
    Main function to run Bayesian Optimization
    """
    # 5 Hyperparameter Domain
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    # Initialize GPyOpt Optimizer
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=create_and_train_model,
        domain=bounds,
        acquisition_type='EI',
        exact_feval=True
    )

    # Run optimization for maximum 30 iterations
    optimizer.run_optimization(max_iter=30)

    # Plot convergence
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')

    # Save report to bayes_opt.txt
    best_x = optimizer.x_opt
    best_y = optimizer.fx_opt

    report_content = (
        "Bayesian Optimization Report\n"
        "============================\n"
        f"Best Learning Rate: {best_x[0]:.6f}\n"
        f"Best Number of Units: {int(best_x[1])}\n"
        f"Best Dropout Rate: {best_x[2]:.4f}\n"
        f"Best L2 Regularization: {best_x[3]:.6f}\n"
        f"Best Batch Size: {int(best_x[4])}\n"
        f"Optimal Validation Error: {best_y:.6f}\n"
        f"Optimal Validation Accuracy: {(1.0 - best_y) * 100:.2f}%\n"
    )

    with open('bayes_opt.txt', 'w') as f:
        f.write(report_content)


if __name__ == '__main__':
    main()
