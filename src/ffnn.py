from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from adamoptimizer import AdamOptimizer


class Value:
    def __init__(self, data, _prev=None, _op=''):
        self.data = data
        self.gradient = 0.0
        self._back = lambda: None
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self._layer : List[Value] = []

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

    def log(self):
        out = Value(np.log(self.data + 1e-7), _prev={self}, _op='log')
        def _back():
            self.gradient += (1 / (self.data + 1e-7)) * out.gradient
        out._back = _back
        return out

    def abs(self):
        out = Value(abs(self.data), _prev={self}, _op='abs')
        def _back():
            self.gradient += (1.0 if self.data > 0 else -1.0) * out.gradient
        out._back = _back
        return out

    def exp(self):
        out = Value(np.exp(self.data), _prev={self}, _op='exp')
        def _back():
            self.gradient += out.data * out.gradient
        out._back = _back
        return out

    def leaky_relu(self, alpha=0.01):
        out = Value(self.data if self.data > 0 else alpha * self.data, _prev={self}, _op='leaky_relu')
        def _back():
            self.gradient += (1.0 if self.data > 0 else alpha) * out.gradient
        out._back = _back
        return out

    def elu(self, alpha=1.0):
        out = Value(self.data if self.data > 0 else alpha * (np.exp(self.data) - 1), _prev={self}, _op='elu')
        def _back():
            self.gradient += (1.0 if self.data > 0 else out.data + alpha) * out.gradient
        out._back = _back
        return out

    def softmax(self):
        layer = self._layer
        exps = [np.exp(neuron.data) for neuron in layer]
        sum_exps = sum(exps)
        proba = [e / sum_exps for e in exps]
        i = layer.index(self)
        out = Value(proba[i], _prev=set(layer), _op='softmax')
        def _back():
            for j, neuron in enumerate(layer):
                if j == i:
                    neuron.gradient += proba[i]*(1-proba[i]) * out.gradient
                else:
                    neuron.gradient += -proba[i]*proba[j] * out.gradient
        out._back = _back
        return out

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

    def clear_graph(self):
        visited = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.extend(node._prev)
            node._back = lambda: None
            node._prev = set()


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_function: str = "", _V=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = []
        self.biases = []
        self._V = _V or Value

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
        elif method == "xavier":
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            w = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
            b = np.random.uniform(-limit, limit, self.output_size)
        elif method == "he":
            stddev = np.sqrt(2 / self.input_size)
            w = np.random.normal(0, stddev, (self.input_size, self.output_size))
            b = np.random.normal(0, stddev, self.output_size)
        else:
            print(f"Unknown weight initialization method '{method}'. Defaulting to zeros.")
            w = np.zeros((self.input_size, self.output_size))
            b = np.zeros(self.output_size)
        self.weights = [[self._V(w[i][j]) for j in range(self.output_size)] for i in range(self.input_size)]
        self.biases = [self._V(b[j]) for j in range(self.output_size)]

    def forward(self, x: List) -> List:
        z = [sum(x[i]*self.weights[i][j] for i in range(self.input_size)) + self.biases[j] for j in range(self.output_size)]
        if self.activation_function == "softmax":
            for node in z:
                node._layer = z
        act = getattr(self._V, self.activation_function)
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
    I_AM_SPEED (False) : use compiled C++ autograd backend if True -- Bool
    """
    def __init__(self, neurons: List[int], activation_function: List[str] = [],
                 loss_function: str = 'mse', learning_rate: float = 0.1, loss_threshold: float = 0.01,
                 epochs: int = 100, seed: int = 42, weight_init: str = "zeros",
                 weight_ub: float = 1.0, weight_lb: float = 0.0, weight_mean: float = 0.0,
                 weight_variance: float = 1.0, batch_size: int = 32, verbose: bool = True,
                 l1: float = 0.0, l2: float = 0.0, optimizer: str = 'sgd',
                 adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8,
                 I_AM_SPEED: bool = False):
        if I_AM_SPEED:
            try:
                from back_prop import Value as CppValue
                self._V = CppValue
                print("I AM SPEED")
            except ImportError:
                self._V = Value
                print("C++ backend not found, compile back_prop.cpp first. Falling back to Python.")
        else:
            self._V = Value

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
        self.X = np.array([])
        self.y = np.array([])
        self.verbose = verbose
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self._adam = AdamOptimizer(alpha=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon) if optimizer == 'adam' else None
        self.input_layer = Layer(neurons[0], neurons[1], self.activation_function[0], _V=self._V)
        self.hidden_layers = [Layer(neurons[i], neurons[i+1], self.activation_function[i], _V=self._V) for i in range(1, len(neurons)-2)]
        self.output_layer = Layer(neurons[-2], neurons[-1], self.activation_function[-1], _V=self._V)
        self.res : List[List] = []
        self.loss = self._V(0.0)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._initialize_weights(weight_init, weight_lb, weight_ub, weight_mean, weight_variance, seed)

    def _initialize_weights(self, method : str, lower_bound: float = 0.0, upper_bound: float = 1.0, mean: float = 0.0, variance : float = 1.0, seed : int = 42) -> None:
        np.random.seed(seed)
        for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
            layer.init_weights(method, lower_bound, upper_bound, mean, variance, seed)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        self.X = X
        self.y = y
        self.history = {"train_loss": [], "val_loss": []}
        self._train(X_val=X_val, y_val=y_val)
        return self.history

    def _compute_validation_loss(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        old_res, old_y = self.res, self.y
        self.res = []
        self.y = np.asarray(y_val)
        for idx in range(X_val.shape[0]):
            x_input = [self._V(X_val[idx, j]) for j in range(X_val.shape[1])]
            self._forward_propagation(x_input)
        loss_val = self._compute_loss(0).data
        self.loss.clear_graph()
        self.res = old_res
        self.y = old_y
        return loss_val

    def _train(self, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        for epoch in range(self.epochs):
            batch = 0
            epoch_loss = 0.0
            n_batches = 0
            while batch*self.batch_size < self.X.shape[0]:
                for i in range(self.batch_size):
                    idx = i + batch * self.batch_size
                    if idx >= self.X.shape[0]:
                        break
                    x_input = [self._V(self.X[idx, j]) for j in range(self.X.shape[1])]
                    self._forward_propagation(x_input)
                self.loss = self._compute_loss(batch * self.batch_size)
                epoch_loss += self.loss.data
                n_batches += 1
                self._backward_propagation()
                batch += 1
                self.res = []
            avg_loss = epoch_loss / n_batches
            self.history["train_loss"].append(avg_loss)
            msg = f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}'
            if X_val is not None and y_val is not None:
                val_loss = self._compute_validation_loss(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                msg += f', Val Loss: {val_loss:.6f}'
            if self.verbose:
                print(msg)

            if avg_loss < self.loss_threshold:
                print('Loss threshold reached. Stopping training.')
                break

    def _forward_propagation(self, x_input) -> None:
        out = self.input_layer.forward(x_input)
        for layer in self.hidden_layers:
            out = layer.forward(out)
        out = self.output_layer.forward(out)
        self.res.append(out)

    def _compute_loss(self, offset: int = 0):
        V = self._V
        total = V(0.0)
        if self.loss_function == 'mse':
            for i in range(len(self.res)):
                y_i = self.y[i+offset]
                for j in range(len(self.res[i])):
                    target = float(y_i[j]) if hasattr(y_i, '__len__') else float(y_i)
                    total += (self.res[i][j] - V(target)) ** 2
            total = total * (1 / len(self.res))
        elif self.loss_function == "binary_cross_entropy":
            for i in range(len(self.res)):
                y_i = self.y[i+offset]
                if not hasattr(y_i, '__len__'):
                    target = [1.0 - float(y_i), float(y_i)]
                else:
                    target = y_i
                for j in range(len(self.res[i])):
                    p = self.res[i][j]
                    y = float(target[j])
                    total += -(V(y) * p.log() + V(1 - y) * (V(1.0) - p).log())
            total = total * (1 / len(self.res))
        elif self.loss_function == "categorical_cross_entropy":
            n_classes = len(self.res[0])
            for i in range(len(self.res)):
                y_i = self.y[i+offset]
                if not hasattr(y_i, '__len__'):
                    target = [1.0 if j == int(y_i) else 0.0 for j in range(n_classes)]
                else:
                    target = y_i
                for j in range(len(self.res[i])):
                    total += -(V(float(target[j])) * self.res[i][j].log())
            total = total * (1 / len(self.res))
        else:
            print(f"Unknown loss function '{self.loss_function}'. Defaulting to MSE.")
            for i in range(len(self.res)):
                for j in range(len(self.res[i])):
                    total += (self.res[i][j] - V(float(self.y[i+offset][j]))) ** 2
            total = total * (1 / len(self.res))
        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for layer in all_layers:
            for row in layer.weights:
                for w in row:
                    if self.l1 > 0:
                        total += V(self.l1) * w.abs()
                    if self.l2 > 0:
                        total += V(self.l2) * w ** 2
        return total

    def _backward_propagation(self) -> None:
        self.loss.back()
        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        if self._adam is not None:
            all_params = []
            all_grads = []
            for layer in all_layers:
                for neuron in layer.weights:
                    for w in neuron:
                        all_params.append(w.data)
                        all_grads.append(w.gradient)
                for bias in layer.biases:
                    all_params.append(bias.data)
                    all_grads.append(bias.gradient)
            updated = self._adam.step(np.array(all_params), np.array(all_grads))
            idx = 0
            for layer in all_layers:
                for neuron in layer.weights:
                    for w in neuron:
                        w.data = updated[idx]
                        w.gradient = 0.0
                        idx += 1
                for bias in layer.biases:
                    bias.data = updated[idx]
                    bias.gradient = 0.0
                    idx += 1
        else:
            for layer in all_layers:
                for neuron in layer.weights:
                    for weight in neuron:
                        weight.data -= self.learning_rate * weight.gradient
                        weight.gradient = 0.0
                for bias in layer.biases:
                    bias.data -= self.learning_rate * bias.gradient
                    bias.gradient = 0.0
        self.loss.clear_graph()

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        old_res, old_y = self.res, self.y
        self.res = []
        self.y = np.asarray(y)
        for idx in range(X.shape[0]):
            x_input = [self._V(X[idx, j]) for j in range(X.shape[1])]
            self._forward_propagation(x_input)
        self.loss = self._compute_loss(0)
        self.loss.back()
        result = self.get_weights()
        self.loss.clear_graph()
        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for layer in all_layers:
            for row in layer.weights:
                for w in row:
                    w.gradient = 0.0
            for b in layer.biases:
                b.gradient = 0.0
        self.res = old_res
        self.y = old_y
        return result

    def get_weights(self) -> List[Dict[str, np.ndarray]]:
        result = []
        all_layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for l_idx, layer in enumerate(all_layers):
            result.append({
                "layer": l_idx,
                "weights": np.array([[w.data for w in row] for row in layer.weights]),
                "biases": np.array([b.data for b in layer.biases]),
                "weight_gradients": np.array([[w.gradient for w in row] for row in layer.weights]),
                "bias_gradients": np.array([b.gradient for b in layer.biases]),
            })
        return result

    def inspect(self, x_row: np.ndarray) -> None:
        self.res = []
        x_input = [self._V(x_row[j]) for j in range(len(x_row))]
        self._forward_propagation(x_input)
        print("Raw output values:", [round(v.data, 6) for v in self.res[-1]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        self.res = []
        for i in range(X.shape[0]):
            x_input = [self._V(X[i, j]) for j in range(X.shape[1])]
            self._forward_propagation(x_input)
            predictions.append([neuron.data for neuron in self.res[-1]])
        arr = np.array(predictions)
        if arr.shape[1] == 1:
            return arr.flatten()
        return np.argmax(arr, axis=1).astype(int)
