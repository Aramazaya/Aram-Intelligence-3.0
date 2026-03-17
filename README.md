# Tugas Besar 1 Pembelajaran Mesin
![Neuron](https://github.com/Aramazaya/Aram-Intelligence-3.0/blob/ee653483010e770825784f6bd00dd56df2cf9fbe/doc/images%20(2).jpg)

Implementation of Feedforward Neural Network (FFNN) from scratch

## Experiments
Experiments within the notebook provided include:
- Layer depth and width
- Hidden Layer Activation Function
- Learning Rate
- Weight initialization
- Regularisation L1 vs L2 vs None
- Comparison with sklearn MLP library
- Comparison of Convergence Speed with Adam Optimizer

## How to setup
All algorithms, dataset, as well as the necessary imports have been put together inside the Jupyter Notebook in `src/Notebook.ipynb` which includes dataset, model, as well as necessary libraries, Exploratory Data Analysis, Data Preprocessing, and the experiments. The implemented FFNN is in the `ffnn.py` file and the implemented adam optimizer is in the `adamoptimizer.py` file.

Simply open the notebook with a supported application or website such as an IDE, Google Colab, Deepnote, etc. and press the `Run All` Button.

When running on an IDE, the necessary libraries will need to be installed beforehand using `pip install`. The libraries needed include:
- sklearn
- pandas
- numpy
- seaborn
- matplotlib

## Contributors
<center>

| Nama | NIM | Pembagian Tugas |
|----------|----------| -- |
| Sebastian Hung Yansen | 13523070 | Laporan, Pengujian, Implementasi Adam Optimizer, Notebook |
| Aramazaya | 13523082 | FFNN model, Autograd, Xavier&He Weight, Leaky ReLU & eLU |

</center>
