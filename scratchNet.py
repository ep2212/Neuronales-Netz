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


class SquaredError(DifferentiableFunction):
    """ Quadratische Fehlerfunktion
        Durch das Quadrieren wird sichergestellt, dass der Fehler nicht negativ wird
        und höhere Differenzen stärker ins Gewicht fallen
    """

    def derivative(self, target, actual):
        return actual - target

    def __call__(self, target, actual):
        return 0.5 * np.sum((target - actual) ** 2)

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

    def prepare_inputs(self, images, labels=None):
        return images if labels is None else images, labels

    def feed_forward_layer(self, input_activations):
        self.layer_inputs = np.dot(
            self.weights, input_activations) + self.biases
        self.activation_vec = self.activation_func(self.layer_inputs)
        return self.activation_vec



    def initialize_parameters(self):
        if self.weights is None: # zufällig erstellte Matrix mit Werten aus der Standardnormalverteilung
            self.weights = np.random.randn(self.neuron_count, self.prev_layer.neuron_count)
        if self.biases is None:
            self.biases = np.random.randn(self.neuron_count, 1)

    def compute_cost_gradients(self, label_vec, cost_func):
        cost_gradients = cost_func.derivative(
            self.activation_vec, label_vec
        ) * self.activation_func.derivative(self.layer_inputs)
        self._update_layer_gradients(cost_gradients)
        return cost_gradients



    def feed_backwards(self, prev_input_gradients):
        new_input_gradients = np.dot(
            self.next_layer.weights.transpose(), prev_input_gradients
        ) * self.activation_func.derivative(self.layer_inputs)
        self._update_layer_gradients(new_input_gradients)
        return new_input_gradients

    def _update_layer_gradients(self, input_gradients):
        self.bias_gradients = input_gradients
        self.weight_gradients = np.dot(
            input_gradients, self.prev_layer.activation_vec.transpose()
        )



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

    def feed_forward_layer(self, input_activations):
        self.activation_vec = input_activations
        return self.activation_vec


    def prepare_inputs(self, images, labels=None):
        flattened_images = images.reshape(images.shape[0], self.neuron_count, 1)
        if labels is not None:
            labels = labels.reshape(labels.shape[0], -1, 1)
            return flattened_images, labels
        return flattened_images



class ScratchNet:

    def __init__(self, layers):  # Konstruktor, dem wir die Layer übergeben
        self.learning_rate = 0.5
        self.cost_func = SquaredError()
        self.layers = layers # hier werden die Layers gespeichert, die übergeben wurden
        for index, layer in enumerate(self.layers): # hier werden die Layers miteinander verkettet
            layer.prev_layer = self.layers[index - 1] if index > 0 else None
            layer.next_layer = (
                self.layers[index + 1] if index + 1 < len(self.layers) else None
            )
            layer.depth = index
            layer.initialize_parameters() # hier werden für jeden Layer die Gewichte und Biase initiiert

    def fit(self, train_images, train_labels, epochs=1, batch_size=1):
        # Preprocessing der Trainingsdaten durch den input-Layer
        train_images, train_labels = self.layers[0].prepare_inputs(
            train_images, train_labels
        )
        training_data = list(zip(train_images, train_labels))
        losses, accuracies = self._gradient_descent(training_data, epochs=epochs)
        return losses, accuracies

    def _gradient_descent(self, training_data, epochs=1):
        losses, accuracies = list(), list()
        for epoch in trange(epochs):
            self._update_parameters(training_data)
            loss, accuracy = (
                self._calculate_loss(training_data),
                self._calculate_accuracy(training_data),
            )
            losses.append(loss)
            accuracies.append(accuracy)
            print(
                "Epoch {0}: loss ={1:.3f} acc={2:.2f}".format(epoch + 1, loss, accuracy)
            )
        return losses, accuracies

    def _update_parameters(self, input_samples):
        weight_gradients = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        bias_gradients = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]

        #Summe aller Gewichts- und Bias Updates
        for sample in input_samples:
            sample_weight_gradients, sample_bias_gradients = self._backpropagate(sample)
            # Addiert zu den Gradienten
            for i in range(len(weight_gradients)):
                weight_gradients[i] += sample_weight_gradients[i]
                bias_gradients[i] += sample_bias_gradients[i]
#            weight_gradients = np.add(weight_gradients, sample_weight_gradients)
#            bias_gradients = np.add(bias_gradients, sample_bias_gradients)

        # Durchschnitt über alle Gewichts- und Bias Updates
        # Der Einfluss der Updates wird durch die 'Learning_rate' beeinflusst
        for layer, layer_weight_gradients, layer_bias_gradients in zip(
            self.layers[1:], weight_gradients, bias_gradients
        ):
            layer.weights += (
                    self.learning_rate * layer_weight_gradients / len(input_samples)
        )
            layer.biases += (
                    self.learning_rate * layer_bias_gradients / len(input_samples)
            )

    def _backpropagate(self, training_sample):
        train_input, train_output = training_sample
        self._feed_forward(train_input)
        # Berechnet die Gradienten des letzten Layer in Abhängigkeit der Kostenfunktion
        gradients = self.layers[-1].compute_cost_gradients(
            train_output, cost_func=self.cost_func
        )

        # Nur für die Hidden Layer werden die Gradienten rückwärts durch das netz propagiert
        for layer in reversed(self.layers[1:-1]):
            # 'gradients' wird mit jedem Layer überschrieben
            gradients = layer.feed_backwards(gradients)

        # Akkumuliert alle Gradienten, die mit Backpropagation berechnet wurden
        weight_gradients = [layer.weight_gradients for layer in self.layers[1:]]
        bias_gradients = [layer.bias_gradients for layer in self.layers[1:]]
        return weight_gradients, bias_gradients

    def predict(self, model_inputs):
        # preprocessing der 'model_inputs' durch den Input-Layer
        model_inputs = self.layers[0].prepare_inputs(model_inputs) # reshapen
        predicted = np.zeros((model_inputs.shape[0], self.layers[-1].neuron_count, 1))
        for i, model_input in enumerate(model_inputs):
            predicted[i] = self._feed_forward(model_input)
        return predicted

    def _feed_forward(self, input_sample):
        for layer in self.layers:
            # input_sample wird mit jedem Layer überschrieben
            input_sample = layer.feed_forward_layer(input_sample)
        return input_sample


    def evaluate(self, validation_images, validation_labels):  # funktioniert wie bei keras
        # Preprocessing der Validierungsdaten durch den Input Layer
        validation_images, validation_labels = self.layers[0].prepare_inputs(
            validation_images, validation_labels
        )
        validation_data = list(zip(validation_images, validation_labels))
        return (
            self._calculate_loss(validation_data),
            self._calculate_accuracy(validation_data),
        )

    def _calculate_loss(self, input_samples):
        total_error = 0.0
        for sample in input_samples:
            image, label_vec = sample
            output_activations = self._feed_forward(image)
            total_error += self.cost_func(label_vec, output_activations)
        return total_error / len(input_samples)

    def _calculate_accuracy(self, input_samples):
        results = [
            (np.argmax(self._feed_forward(image)), np.argmax(expected_label))
            for image, expected_label in input_samples
        ]
        num_correct = sum(int(x == y) for (x, y) in results)
        return num_correct / len(input_samples)

    def compile(self, learning_rate=None, loss=SquaredError()):  # das Netz parametrieren
        # soll einfach die definierten Werte im Netz speichern
        self.learning_rate = learning_rate or self.learning_rate
        self.cost_func = loss or self.cost_func


    def inspect(self):  # Struktur des netes ausdrucken
        print(f"---------- {self.__class__.__name__} ----------")
        print(f"  # Inputs: {self.layers[0].neuron_count}")
        for layer in self.layers:
            layer.inspect()




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

    # predicted = model.predict(np.array([[0, 0]]))
    # print(predicted)
    # return

    # model.compile(learning_rate=0.1, loss=SquaredError()) # kein Optimizer, nur ein Lernverfahren
    model.compile(learning_rate=1.0, loss=SquaredError()) # kein Optimizer, nur ein Lernverfahren

    start = time.time()
    losses, accuracies = model.fit(
    #    train_inputs, train_vec_labels, epochs=4,  batch_size=4)
         train_inputs, train_vec_labels, epochs=20,  batch_size=4)
    end = time.time()
    print('Trainingsdauer: {:.1f}s'.format(end - start))

    val_loss, val_accuracy = model.evaluate(train_inputs, train_vec_labels
    )
    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {val_accuracy}")

    plot_loss_and_accuracy(losses, accuracies, xlabel="epochs")



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
