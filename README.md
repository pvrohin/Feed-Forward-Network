# Feed-Forward Neural Network

Implementation of a feed-forward neural network for multi-class classification on the Fashion-MNIST dataset. This project includes implementations of forward propagation, backpropagation, gradient descent optimization, and visualization utilities.

## Overview

This repository contains implementations of fully connected feed-forward neural networks with the following features:

- **Multi-layer neural networks** with configurable architecture
- **ReLU activation** functions for hidden layers
- **Softmax activation** for output layer
- **Stochastic Gradient Descent (SGD)** optimization
- **L2 regularization** to prevent overfitting
- **Gradient checking** for verification
- **Hyperparameter tuning** capabilities
- **PCA-based visualization** of SGD trajectory

## Repository Structure

```
.
├── README.md                    # This file
├── ques4.py                     # Basic neural network implementation
├── ques_4_template.py          # Comprehensive template with hyperparameter search
├── ques_4_template_2.py        # Alternative template
├── plot_sgd.py                 # SGD trajectory visualization script
├── homework4.pdf               # Assignment instructions
└── Figure_1.png               # Reference figure
```

## Features

### Neural Network Components

1. **Forward Propagation**
   - Linear transformations with weight matrices and bias vectors
   - ReLU activation for hidden layers
   - Softmax activation for the output layer
   - Cross-entropy loss with L2 regularization

2. **Backpropagation**
   - Efficient gradient computation for all layers
   - Support for ReLU and softmax derivatives
   - Regularization term in gradient updates

3. **Optimization**
   - Mini-batch Stochastic Gradient Descent
   - Configurable learning rate (epsilon)
   - Support for learning rate scheduling
   - Early stopping based on validation loss

4. **Hyperparameter Search**
   - Automated grid search over multiple hyperparameters
   - Validation set evaluation
   - Best model selection based on validation performance

## Requirements

### Dependencies

```bash
numpy
matplotlib
scipy
scikit-learn
numba
```

Install dependencies using:

```bash
pip install numpy matplotlib scipy scikit-learn numba
```

### Data Files

The code expects the following Fashion-MNIST data files in the working directory:
- `fashion_mnist_train_images.npy`
- `fashion_mnist_train_labels.npy`
- `fashion_mnist_test_images.npy`
- `fashion_mnist_test_labels.npy`

## Usage

### Basic Training (`ques4.py`)

Simple implementation for training a neural network:

```bash
python ques4.py
```

This script:
- Loads and preprocesses Fashion-MNIST data
- Splits training data into training and validation sets (80/20 split)
- Trains a 4-layer network with 30 units per hidden layer
- Prints training loss during epochs

### Advanced Training with Hyperparameter Search (`ques_4_template.py`)

Comprehensive training with automated hyperparameter tuning:

```bash
python ques_4_template.py
```

Key features:
- Gradient checking verification
- Hyperparameter grid search
- Validation loss tracking
- Test set evaluation
- Accuracy reporting

### Configuration

You can modify hyperparameters in the script:

```python
NUM_HIDDEN_LAYERS = 3        # Number of hidden layers
NUM_HIDDEN = 30              # Units per hidden layer
NUM_OUTPUT = 10              # Output classes (Fashion-MNIST)
ALPHA = 0.001                # L2 regularization coefficient
EPSILON = 1                  # Learning rate
NUM_EPOCHS = 3               # Number of training epochs
MINI_BATCH_SIZE = 64         # Mini-batch size for SGD
```

### SGD Trajectory Visualization (`plot_sgd.py`)

Visualize the optimization trajectory in a reduced 2D space using PCA:

```python
from plot_sgd import plotSGDPath

# After training, visualize trajectory
plotSGDPath(X_tr, y_tr, trajectory)
```

## Architecture

The neural network architecture consists of:

1. **Input Layer**: 784 neurons (28×28 Fashion-MNIST images, flattened)
2. **Hidden Layers**: Configurable number of layers with ReLU activation
3. **Output Layer**: 10 neurons with softmax activation (for 10 Fashion-MNIST classes)

### Activation Functions

- **ReLU** (Hidden layers): `f(x) = max(0, x)`
- **Softmax** (Output layer): `f(x_i) = exp(x_i) / Σ exp(x_j)`

### Loss Function

Cross-entropy loss with L2 regularization:

```
L = -1/n Σ y*log(ŷ) + α/(2n) Σ ||W||²
```

Where:
- `n` = number of training examples
- `α` = regularization coefficient
- `W` = all weight matrices

## Implementation Details

### Weight Initialization

Weights are initialized using:
- Normal distribution with zero mean
- Standard deviation of `1/√(units_in_previous_layer)`
- Biases initialized with small random values

### Data Preprocessing

- Pixel values normalized to [0, 1] range (division by 255)
- Labels converted to one-hot encoding
- Data shuffled before train/validation split

### Gradient Descent

- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Batch processing**: Mini-batch updates
- **Learning rate**: Fixed or scheduled (optional)

## Model Class (`ques4.py`)

The `Model_architeture` class provides an object-oriented interface:

```python
Hypset = [layers, units_per_layer, alpha, epsilon]
model = Model_architeture(Hypset, X_tr, y_tr, X_val, y_val)

for epoch in range(num_epochs):
    model.forward_propagation()
    loss = model.loss()
    gradW, gradb = model.back_propagate()
    model.parameter_update(gradW, gradb)
```

## Results

After training, the model reports:
- Training loss
- Validation loss
- Test set cost
- Classification accuracy

## Notes

- The implementation uses NumPy for all computations
- Gradient checking is available to verify backpropagation correctness
- PCA visualization reduces high-dimensional weight space to 2D for plotting
- Random seeds are set for reproducibility

## License

See the LICENSE file for details.
