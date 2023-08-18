# Activation Functions

## Overview

`activations` provides a suite of activation functions commonly employed in the context of neural networks. 
In addition to the functional implementation, this module offers gradient computation for each activation function, 
aiding in the understanding and application of backpropagation. 
The `softmax` function is an exception, as it inherently involves multiple inputs and outputs, making its gradient 
computation distinct.

## Table of Contents

- [Sigmoid](#sigmoid)
- [Tanh](#tanh)
- [ReLU](#relu)
- [Swish](#swish)
- [CELU](#celu)
- [Softmax](#softmax)

## Sigmoid

**Location**: Defined in `sigmoid.py`

**Formula**: 

![](https://quicklatex.com/cache3/7a/ql_668dc0601663ef0c48396ed1ac15617a_l3.png)

The sigmoid function compresses its input to the range between 0 and 1.

**Gradient**:

![](https://quicklatex.com/cache3/1e/ql_bda49ddfef38d523f27009a45c28b61e_l3.png)

This gradient results in a vanishing gradient problem for values that are very high or very low.

## Tanh

**Location**: Defined in `tanh.py`

**Formula**:

![](https://quicklatex.com/cache3/9f/ql_72eebf5038a7f863b236caa209a7b09f_l3.png)

The tanh function compresses its input to the range between -1 and 1.

**Gradient**:

![](https://quicklatex.com/cache3/5a/ql_1aec5bd7f300437b4a6a9f9342a6165a_l3.png)

## ReLU

**Location**: Defined in `relu.py`

**Formula**: 

![](https://quicklatex.com/cache3/a0/ql_37999e1feff124baca2413a754b3bfa0_l3.png)

ReLU introduces non-linearity into the model without affecting the receptive fields of convolutions.

**Gradient**: 
\[ 
f'(x) = 
\begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{otherwise} 
\end{cases}
\]

This makes it computationally efficient but can result in dead neurons during training.

## Swish

**Location**: Defined in `swish`

**Formula**: \[ f(x) = x \times sigmoid(\beta x) \] where \( \beta \) is a trainable parameter.

Swish is a self-gated function, attempting to balance the advantages of ReLU and sigmoid.

**Gradient**: The gradient is complex due to the presence of \( \beta \) and is often computed using automatic differentiation tools.

## CELU

**Location**: Defined in `celu`

**Formula**: 
\[ 
f(x) = 
\begin{cases} 
x & \text{if } x > 0 \\
\alpha \times (e^{\frac{x}{\alpha}} - 1) & \text{otherwise} 
\end{cases}
\]

CELU is a variant of the exponential linear units (ELU) with a controllable scaling parameter \( \alpha \).

**Gradient**: The gradient has different expressions for positive and negative values of x, with the negative portion involving \( \alpha \).

## Softmax

**Location**: Defined in `softmax`

**Formula**: \[ f(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} \]

The softmax function is used in multi-class classification problems. It converts input into a probability distribution over multiple classes.

**Gradient**: The gradient of softmax is more intricate due to the normalization involved, and its computation requires the Jacobian matrix.
