import abc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import time

from fontTools.misc.cython import returns
from tqdm import tqdm, trange
from deeplearning2020 import helpers

def xor_keras():
    """Beispielnetz für das XOR-Problem"""
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])

    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(2, 1)),
            tf.keras.layers.Dense(4, activation='?'),
            tf.keras.layers.Dense(2, activation='?')
                    ]
    )
    model.compile(optimizer='?', loss='?', metrics=['?'])

    model.fit(train_inputs, train_vec_labels, epochs=20)

    val_loss, val_accuracy = model.evaluate(train_inputs, train_vec_labels, verbose=False)

    print("Validation loss: %.2f" % val_loss)
    print("Validation accuracy: %.2f" % val_accuracy)

    return model
