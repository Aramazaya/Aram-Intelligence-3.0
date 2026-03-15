from typing import List
import pandas as pd
import numpy as np


class Value:
    def __init__(self, data, _prev=None, _op=''):
        self.data = data
        self.gradient = 0.0
        self._back = lambda: None
        self._prev = _prev if _prev is not None else set()
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
        return
    
    def back(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.gradient = 1.0
        for node in reversed(topo):
            node._back()


class Layer:
    def __init__ (self, input_size: int, output_size: int, activation_function: str = ""):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = []
        self.biases = []

    def init_weights(self, method: str, lower_bound: float = 0.0, upper_bound: float = 1.0, mean: float = 0.0, variance: float = 1.0, seed: int = 42) -> None:
        if method == "zeros":
            w = np.zeros((self.input_size, self.output_size))
            b = np.zeros(self.output_size)
        elif method == "random":
            w = np.random.uniform(lower_bound, upper_bound, (self.input_size, self.output_size))
            b = np.random.uniform(lower_bound, upper_bound, self.output_size)
        elif method == "normal":
            w = np.random.normal(mean, variance, (self.input_size, self.output_size))
            b = np.random.normal(mean, variance, self.output_size)
        else:
            print(f"Unknown weight initialization method '{method}'. Defaulting to zeros.")
            w = np.zeros((self.input_size, self.output_size))
            b = np.zeros(self.output_size)
        self.weights = [[Value(w[i][j]) for j in range(self.output_size)] for i in range(self.input_size)]
        self.biases = [Value(b[j]) for j in range(self.output_size)]

    def forward(self, x: List[Value]) -> List[Value]:
        z = [sum(x[i]*self.weights[i][j] for i in range(self.input_size)) + self.biases[j] for j in range(self.output_size)]
        act = getattr(Value, self.activation_function)
        return [act(val) for val in z]


class FeedForwardNN:
    """Main class for the FFNN.

    Parameters
    ----------
    neurons : num of neurons per each layer -- Int List
    activation_function (linear) : activation function for each layer starting with first hidden layer-- String List
    loss_function (mse) : loss function to optimize -- String
    learning_rate (0.1) : learning rate for gradient descent -- Float
    loss_threshold (0.01) : threshold for early stopping -- Float
    epochs (100) : maximum number of epochs -- Int
    seed (42) : random seed for weight initialization -- Int
    weight_init (zeros) : method for weight initialization (zeros, random, normal) default zeros -- String
    weight_ub (1.0) : upper bound for random weight initialization -- Float
    weight_lb (0.0) : lower bound for random weight initialization -- Float
    weight_mean (0.0) : mean for normal weight initialization -- Float
    weight_variance (1.0) : variance for normal weight initialization -- Float
    """
    def __init__(self, neurons: List[int], activation_function: List[str] = [], 
                 loss_function: str = 'mse', learning_rate: float = 0.1, loss_threshold: float = 0.01, 
                 epochs: int = 100, seed: int = 42, weight_init: str = "zeros", 
                 weight_ub: float = 1.0, weight_lb: float = 0.0, weight_mean: float = 0.0, 
                 weight_variance: float = 1.0, batch_size: int = 32):
        self.neurons = neurons
        if not activation_function:
            self.activation_function = ['linear' for _ in range(len(neurons) - 1)]
        elif len(activation_function) != len(neurons) - 1:
            self.activation_function = activation_function + ['linear'] * (len(neurons) - 1 - len(activation_function))
        else:
            self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.epochs = epochs
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.batch_size = batch_size
        self.input_layer = Layer(neurons[0], neurons[1], self.activation_function[0])
        self.hidden_layers = [Layer(neurons[i], neurons[i+1], self.activation_function[i]) for i in range(1, len(neurons)-2)]
        self.output_layer = Layer(neurons[-2], neurons[-1], self.activation_function[-1])
        self.res : List[List[Value]] = []
        self.loss = Value(0.0)
        self._initialize_weights(weight_init, weight_lb, weight_ub, weight_mean, weight_variance, seed)

    pd.DataFrame
    def _initialize_weights(self, method : str, lower_bound: float = 0.0, upper_bound: float = 1.0, mean: float = 0.0, variance : float = 1.0, seed : int = 42) -> None:
        np.random.seed(seed)
        for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
            layer.init_weights(method, lower_bound, upper_bound, mean, variance, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y
        self._train()

    def _train(self) -> None:
        for epoch in range(self.epochs):
            batch = 0
            while batch*self.batch_size < self.X.shape[0]:
                for i in range(self.batch_size):
                    idx = i + batch * self.batch_size
                    if idx >= self.X.shape[0]:
                        break
                    x_input = [Value(self.X.iloc[idx, j]) for j in range(self.X.shape[1])]
                    self._forward_propagation(x_input)
                self.loss = self._compute_loss(batch * self.batch_size)
                self._backward_propagation()
                batch +=1
                self.res = []
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {self.loss}')
            if self.loss.data < self.loss_threshold:
                print('Loss threshold reached. Stopping training.')
                break

    def _forward_propagation(self, x_input) -> None:
        out = self.input_layer.forward(x_input)
        for layer in self.hidden_layers:
            out = layer.forward(out)
        out = self.output_layer.forward(out)
        self.res.append(out)

    def _compute_loss(self, offset: int = 0) -> Value:
        total = Value(0.0)
        if self.loss_function == 'mse':
            for i in range(len(self.res)):
                for j in range (len(self.res[i])):
                    total += (self.res[i][j] - Value(self.y.iloc[i+offset])) ** 2
            return total * (1 / len(self.res))
        elif self.loss_function == "binary_cross_entropy":
            for i in range(len(self.res)):
                for j in range (len(self.res[i])):
                    p = self.res[i][j].sigmoid()
                    total += -(Value(self.y.iloc[i+offset]) * p) - (Value(1 - self.y.iloc[i+offset]) * (1 - p))
            return total * (1 / len(self.res))
        elif self.loss_function == "categorical_cross_entropy":
            for i in range(len(self.res)):
                for j in range (len(self.res[i])):
                    total += -(Value(self.y.iloc[i+offset]) * self.res[i][j])
            return total * (1 / len(self.res))
        else:
            print(f"Unknown loss function '{self.loss_function}'. Defaulting to MSE.")
            for i in range(len(self.res)):
                for j in range (len(self.res[i])):
                    total += (self.res[i][j] - Value(self.y.iloc[i+offset])) ** 2
            return total * (1 / len(self.res))

    def _backward_propagation(self) -> None:
        self.loss.back()
        for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
            for neuron in layer.weights:
                for weight in neuron:
                    weight.data -= self.learning_rate * weight.gradient
                    weight.gradient = 0.0
            for bias in layer.biases:
                bias.data -= self.learning_rate * bias.gradient
                bias.gradient = 0.0
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for i in range(X.shape[0]):
            x_input = [Value(X.iloc[i, j]) for j in range(X.shape[1])]
            self._forward_propagation(x_input)
            predictions.append([neuron.data for neuron in self.res[-1]])
        return np.array(predictions)