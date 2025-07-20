import abc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import time

from fontTools.misc.cython import returns
from tqdm import tqdm, trange
from deeplearning2020 import helpers

def plot_loss_and_accuracy(losses, accuracies, xlabel):
    plt.plot(losses, label='loss')
    plt.plot(accuracies, label='accuracy')
    plt.legend(loc="upper left")
    plt.xlabel(xlabel)
    plt.ylim(top=1, bottom=0)
    plt.show()


class SquaredError:
    """
    Berechnet den mittleren quadratischen Fehler (Mean Squared Error) und dessen Ableitung.
    """

    def __call__(self, y_pred, y_true):
        """
        Macht die Klasse aufrufbar wie eine Funktion, um den Fehler zu berechnen.
        """
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred, y_true):
        """
        Berechnet die Ableitung der Kostenfunktion.
        Wird für die Backpropagation benötigt.
        """
        num_samples = y_true.shape[0] if y_true.ndim > 0 else 1
        return 2 * (y_pred - y_true) / num_samples

class DifferentiableFunction(abc.ABC):
    """ Abstrakte Klasse einer differenzierbaren Funktion
        Die Implementierung der Methoden erfolgt durch Spezialisierung
    """

    def derivative(self, net_input):
        pass

    def __call__(self, net_input):
        pass

class Sigmoid(DifferentiableFunction):

    def derivative(self, net_input):
        return self(net_input) * (1 - self(net_input))

    def __call__(self, net_input):
        return 1 / (1 + np.exp(-net_input))

class ScratchNet:

    def __init__(self, layers):  # Konstruktor, dem wir die Layer übergeben
        self.learning_rate = 0.5
        self.cost_func = SquaredError()
        self.layers = layers # hier werden die Layers gespeichert, die übergeben wurden
        for index, layer in enumerate(self.layers): # hier werden die Layers miteinander verkettet
            layer.prev_layer = self.layers[index - 1] if index > 0 else None
            layer.next_layer = (
                self.layers[index + 1] if index + 1 < len(self.layers) - 1 else None
            )
            layer.depth = index
            layer.initialize_parameters() # hier werden für jeden Layer die Gewichte und Biase initiiert

    def fit(self, train_images, train_labels, epochs=1):
        raise NotImplementedError()

    def predict(self, model_inputs):
        raise NotImplementedError()

    def evaluate(self, validation_images, validation_labels):  # funktioniert wie bei keras
        raise NotImplementedError()

    def compile(self, learning_rate=None, los=None):  # das Netz parametrieren
        raise NotImplementedError()

    def inspect(self):  # Struktur des netes ausdrucken
        print(f"---------- {self.__class__.__name__} ----------")
        print(f"  # Inputs: {self.layers[0].neuron_count}")
        for layer in self.layers:
            layer.inspect()

class DenseLayer:
    def __init__(
            self,
            neuron_count,
            depth=None,
            activation=None,
            biases=None,
            weights=None,
            prev_layer =None,
            next_layer =None,
    ):
        self.depth = depth
        self.next_layer = next_layer
        self.prev_layer = prev_layer

        self.neuron_count = neuron_count
        self.activation_func = activation or Sigmoid()

        self.weights = weights
        self.biases = biases

    def initialize_parameters(self):
        if self.weights is None: # zufällig erstellte Matrix mit Werten aus der Standardnormalverteilung
            self.weights = np.random.randn(self.neuron_count, self.prev_layer.neuron_count)
        if self.biases is None:
            self.biases = np.random.randn(self.neuron_count, 1)

    def inspect(self):
        print(f"---------- Layer L={self.depth} ----------")
        print(f" # Neuronen: {self.neuron_count}")
        for n in range(self.neuron_count):
            print(f"  Neuron {n}:")
            if self.prev_layer:
                for w in self.weights[n]:
                    print(f"    Weight: {w}")
                print(f"    Bias: {self.biases[n][0]}")




class FlattenLayer(DenseLayer): # Spezialisierung des DenseLayers, normalerweise nur Input-Layer
    """ initialisierung wird überschrieben mit weniger Parameter,
    damit sollen die Anzahl der Neuronen selbst berechnet werden"""
    def __init__(self, input_shape):
        total_input_neurons =1
        # Beispiele:
        # (28,28) wird 28*28=784
        # (28,28,1) wird 28*28*1=784
        # (4,4,2) wird 4*4*2=32
        for dim in input_shape: # input_shape wird in flache Version gebracht
            total_input_neurons *= dim
        super().__init__(total_input_neurons) # Konstruktor der Superklasse aufrufen

    def initialize_parameters(self): # keine Weights und Biases in einer Flatten-Layer
        pass


def xor():
    """Beispielnetz für das XOR-Problem"""
    train_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_labels = np.array([[0], [1], [1], [0]])

    total_classes = 2
    train_vec_labels = tf.keras.utils.to_categorical(train_labels, total_classes)

    model = ScratchNet(
        [
            FlattenLayer(input_shape=(2, 1)),
            DenseLayer(4, activation=Sigmoid()),
            DenseLayer(2, activation=Sigmoid()),
        ]
    )

    # Wiederholt die Werte, um weniger Epochen trainieren zu müssen
    repeat = (10000, 1)
    train_inputs = np.tile(train_inputs, repeat)
    train_vec_labels = np.tile(train_vec_labels, repeat)

    model.inspect()
    return

    model.compile(learning_rate=0.1, loss=SquaredError()) # kein Optimizer, nur ein Lernverfahren

    start = time.time()
    losses, accuracies = model.fit(
        train_inputs, train_vec_labels, epochs=4)
    end = time.time()
    print('Trainingsdauer: {:.1f}s'.format(end - start))

    val_loss, val_accuracy = model.evaluate(train_inputs, train_vec_labels
    )
    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {val_accuracy}")

    plot_loss_and_accuracy(losses, accuracies, xlabels="epochs")



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

xor()
