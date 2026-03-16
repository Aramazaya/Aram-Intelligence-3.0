#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_set>
#include <functional>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>

namespace py = pybind11;

struct Value : std::enable_shared_from_this<Value> {
    double data;
    double gradient;
    std::string _op;
    std::vector<std::shared_ptr<Value>> _prev;
    std::function<void()> _back_fn;
    std::vector<std::shared_ptr<Value>> _layer;

    Value(double data,
          std::vector<std::shared_ptr<Value>> prev = {},
          std::string op = "")
        : data(data), gradient(0.0), _op(op),
          _prev(std::move(prev)), _back_fn([](){}) {}

    // ------------------------------------------------------------------ ops

    std::shared_ptr<Value> add(std::shared_ptr<Value> other) {
        auto out = std::make_shared<Value>(
            data + other->data,
            std::vector<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, other, w]() {
            if (auto o = w.lock()) {
                self->gradient  += o->gradient;
                other->gradient += o->gradient;
            }
        };
        return out;
    }

    std::shared_ptr<Value> mul(std::shared_ptr<Value> other) {
        double sd = data, od = other->data;
        auto out = std::make_shared<Value>(
            sd * od,
            std::vector<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, other, w, sd, od]() {
            if (auto o = w.lock()) {
                self->gradient  += od * o->gradient;
                other->gradient += sd * o->gradient;
            }
        };
        return out;
    }

    std::shared_ptr<Value> pow_op(double exp) {
        double base = data;
        auto out = std::make_shared<Value>(
            std::pow(base, exp),
            std::vector<std::shared_ptr<Value>>{shared_from_this()}, "**");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, base, exp]() {
            if (auto o = w.lock())
                self->gradient += exp * std::pow(base, exp - 1.0) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> neg() {
        return mul(std::make_shared<Value>(-1.0));
    }

    std::shared_ptr<Value> sub(std::shared_ptr<Value> other) {
        return add(other->neg());
    }

    // --------------------------------------------------------- activations

    std::shared_ptr<Value> sigmoid() {
        double s = 1.0 / (1.0 + std::exp(-data));
        auto out = std::make_shared<Value>(s, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "sigmoid");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, s]() {
            if (auto o = w.lock())
                self->gradient += s * (1.0 - s) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> relu() {
        double val = data > 0.0 ? data : 0.0;
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "relu");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w]() {
            if (auto o = w.lock())
                self->gradient += (o->data > 0.0 ? 1.0 : 0.0) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> tanh_op() {
        double t = std::tanh(data);
        auto out = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "tanh");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, t]() {
            if (auto o = w.lock())
                self->gradient += (1.0 - t * t) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> linear() {
        auto out = std::make_shared<Value>(data, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "linear");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w]() {
            if (auto o = w.lock())
                self->gradient += o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> log_op() {
        double val = std::log(data + 1e-7);
        double d   = data;
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "log");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, d]() {
            if (auto o = w.lock())
                self->gradient += (1.0 / (d + 1e-7)) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> abs_op() {
        double val = std::abs(data);
        double d   = data;
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "abs");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, d]() {
            if (auto o = w.lock())
                self->gradient += (d > 0.0 ? 1.0 : -1.0) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> exp_op() {
        double val = std::exp(data);
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "exp");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w]() {
            if (auto o = w.lock())
                self->gradient += o->data * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> leaky_relu(double alpha = 0.01) {
        double val = data > 0.0 ? data : alpha * data;
        double d   = data;
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "leaky_relu");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, d, alpha]() {
            if (auto o = w.lock())
                self->gradient += (d > 0.0 ? 1.0 : alpha) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> elu(double alpha = 1.0) {
        double val = data > 0.0 ? data : alpha * (std::exp(data) - 1.0);
        double d   = data;
        auto out = std::make_shared<Value>(val, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "elu");
        auto self = shared_from_this();
        std::weak_ptr<Value> w = out;
        out->_back_fn = [self, w, d, alpha]() {
            if (auto o = w.lock())
                self->gradient += (d > 0.0 ? 1.0 : o->data + alpha) * o->gradient;
        };
        return out;
    }

    std::shared_ptr<Value> softmax() {
        if (_layer.empty())
            throw std::runtime_error("_layer not set before calling softmax()");

        int i = -1;
        for (int k = 0; k < (int)_layer.size(); k++)
            if (_layer[k].get() == this) { i = k; break; }
        if (i < 0) throw std::runtime_error("Value not found in its own _layer");

        std::vector<double> exps_vec;
        double sum_exps = 0.0;
        for (auto& n : _layer) {
            double e = std::exp(n->data);
            exps_vec.push_back(e);
            sum_exps += e;
        }
        std::vector<double> proba;
        for (double e : exps_vec) proba.push_back(e / sum_exps);

        auto out = std::make_shared<Value>(
            proba[i],
            std::vector<std::shared_ptr<Value>>(_layer.begin(), _layer.end()),
            "softmax");

        std::vector<std::shared_ptr<Value>> layer_copy = _layer;
        double pi = proba[i];
        std::weak_ptr<Value> w = out;
        out->_back_fn = [layer_copy, w, i, pi, proba]() {
            if (auto o = w.lock()) {
                for (int j = 0; j < (int)layer_copy.size(); j++) {
                    if (j == i)
                        layer_copy[j]->gradient += pi * (1.0 - pi) * o->gradient;
                    else
                        layer_copy[j]->gradient += -pi * proba[j] * o->gradient;
                }
            }
        };
        return out;
    }

    // -------------------------------------------------- backward + cleanup

    void back() {
        std::vector<Value*> topo;
        std::unordered_set<Value*> visited;

        std::function<void(Value*)> build = [&](Value* v) {
            if (!visited.count(v)) {
                visited.insert(v);
                for (auto& c : v->_prev) build(c.get());
                topo.push_back(v);
            }
        };

        build(this);
        gradient = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_back_fn();
    }

    void clear_graph() {
        std::unordered_set<Value*> visited;
        std::vector<std::shared_ptr<Value>> stack = {shared_from_this()};
        while (!stack.empty()) {
            auto sp = stack.back(); stack.pop_back();
            Value* node = sp.get();
            if (visited.count(node)) continue;
            visited.insert(node);
            auto children = node->_prev;  // copy to keep children alive while we clear
            node->_back_fn = [](){};
            node->_prev.clear();
            for (auto& c : children) stack.push_back(c);
        }
    }
};

// ---------------------------------------------------------------- bindings

PYBIND11_MODULE(back_prop, m) {
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<double>())
        .def_readwrite("data",     &Value::data)
        .def_readwrite("gradient", &Value::gradient)
        .def_readwrite("_op",      &Value::_op)
        .def_readwrite("_prev",    &Value::_prev)
        .def_readwrite("_layer",   &Value::_layer)

        // arithmetic
        .def("__add__",  [](std::shared_ptr<Value> s, py::object o) {
            return s->add(py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>()));
        })
        .def("__radd__", [](std::shared_ptr<Value> s, py::object o) {
            return s->add(py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>()));
        })
        .def("__mul__",  [](std::shared_ptr<Value> s, py::object o) {
            return s->mul(py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>()));
        })
        .def("__rmul__", [](std::shared_ptr<Value> s, py::object o) {
            return s->mul(py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>()));
        })
        .def("__pow__",  [](std::shared_ptr<Value> s, double e)  { return s->pow_op(e); })
        .def("__neg__",  [](std::shared_ptr<Value> s)            { return s->neg(); })
        .def("__sub__",  [](std::shared_ptr<Value> s, py::object o) {
            return s->sub(py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>()));
        })
        .def("__rsub__", [](std::shared_ptr<Value> s, py::object o) {
            auto other = py::isinstance<Value>(o)
                ? o.cast<std::shared_ptr<Value>>()
                : std::make_shared<Value>(o.cast<double>());
            return other->sub(s);
        })

        // activations
        .def("sigmoid",    [](std::shared_ptr<Value> s) { return s->sigmoid(); })
        .def("relu",       [](std::shared_ptr<Value> s) { return s->relu(); })
        .def("tanh",       [](std::shared_ptr<Value> s) { return s->tanh_op(); })
        .def("linear",     [](std::shared_ptr<Value> s) { return s->linear(); })
        .def("log",        [](std::shared_ptr<Value> s) { return s->log_op(); })
        .def("abs",        [](std::shared_ptr<Value> s) { return s->abs_op(); })
        .def("exp",        [](std::shared_ptr<Value> s) { return s->exp_op(); })
        .def("leaky_relu", [](std::shared_ptr<Value> s, double a) { return s->leaky_relu(a); },
             py::arg("alpha") = 0.01)
        .def("elu",        [](std::shared_ptr<Value> s, double a) { return s->elu(a); },
             py::arg("alpha") = 1.0)
        .def("softmax",    [](std::shared_ptr<Value> s) { return s->softmax(); })

        // backward
        .def("back",        [](std::shared_ptr<Value> s) { s->back(); })
        .def("clear_graph", [](std::shared_ptr<Value> s) { s->clear_graph(); });
}
