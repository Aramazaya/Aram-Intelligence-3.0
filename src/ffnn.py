from typing import List
import pandas as pd
import numpy as np


class Value:
    def __init__(self, data, _prev=set(), _op=''):
        self.data = data
        self.gradient = 0.0
        self._back = lambda: None
        self._prev = _prev
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _prev={self, other}, _op='+')
        def _back():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._back = _back
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _prev={self, other}, _op='*')
        def _back():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out._back = _back
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other, _prev={self}, _op='**')
        def _back():
            self.gradient += other * (self.data ** (other - 1)) * out.gradient
        out._back = _back
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, _prev={self}, _op='sigmoid')
        def _back():
            self.gradient += s * (1 - s) * out.gradient
        out._back = _back
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, _prev={self}, _op='relu')
        def _back():
            self.gradient += (out.data > 0) * out.gradient
        out._back = _back
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, _prev={self}, _op='tanh')
        def _back():
            self.gradient += (1 - t ** 2) * out.gradient
        out._back = _back
        return out
    
    def linear(self):
        out = Value(self.data, _prev={self}, _op='linear')
        def _back():
            self.gradient += out.gradient
        out._back = _back
        return out
    
    def softmax(self):


class Layer:
    def __init__ (self, input_size: int, output_size: int, activation_function: str = ""):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = []
        self.biases = []

    def _init_weights(self):

class FeedForwardNN:

    def __init__(self, neurons: List[int], activation_function: List[str], 
                 loss_function: str, learning_rate: float, loss_threshold: float, 
                 epochs: int, seed: int = 42, weight_init: str = "zeros", 
                 weight_ub: float = 1.0, weight_lb: float = 0.0, weight_mean: float = 0.0, 
                 weight_variance: float = 1.0):
        """Initializes the FeedForward Neural Network.
        
        Args:
        neurons : num of neurons per each layer -- Int List
        activation_function : activation function for each layer starting with first hidden layer-- String List
        loss_function : loss function to optimize -- String
        learning_rate : learning rate for gradient descent -- Float
        loss_threshold : threshold for early stopping -- Float
        epochs : maximum number of epochs for training -- Int
        seed : random seed for weight initialization -- Int
        weight_init : method for weight initialization (zeros, random, normal) default zeros -- String
        weight_ub : upper bound for random weight initialization -- Float
        weight_lb : lower bound for random weight initialization -- Float
        weight_mean : mean for normal weight initialization -- Float
        weight_variance : variance for normal weight initialization -- Float
        """
        self.neurons = neurons
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.epochs = epochs
        self.X = None
        self.y = None
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.weights = []
        self.weight_grads = []
        self.input_layer = Layer(neurons[0], neurons[1])
        self.hidden_layers = [Layer(neurons[i], neurons[i+1], activation_function[i]) for i in range(1, len(neurons)-2)]
        self.output_layer = Layer(neurons[-2], neurons[-1], activation_function[-1])
        self.res = []
        self.loss = 0.0
        self._initialize_weights('zeros')

    
    def _initialize_weights(self, method : str, lower_bound: float = 0.0, upper_bound: float = 1.0, mean: float = 0.0, variance : float = 1.0, seed : int = 42) -> None:
        np.random.seed(seed)
        

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y
        self._train()

    def _train(self) -> None:
        for epoch in range(self.epochs):
            self.res = self._forward_propagation()
            self.loss = self._compute_loss()
            self._backward_propagation()
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {self.loss}')
            if self.loss < self.loss_threshold:
                print('Loss threshold reached. Stopping training.')
                break

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        return np.zeros(X.shape[0])