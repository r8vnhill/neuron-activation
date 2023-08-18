# Activation Functions

## Overview

The `activations` module provides an array of widely-used activation functions suitable for neural network 
architectures.
Besides the functional expressions, this module meticulously computes the gradient for each function, serving as a
crucial tool in the comprehension and application of the backpropagation algorithm.
The exception to this is the `softmax` function, given its unique nature of handling multiple inputs and outputs which
leads to a distinct gradient evaluation.

## Table of Contents

- [Sigmoid](#sigmoid)
- [Tanh](#tanh)
- [ReLU](#relu)
- [Swish](#swish)
- [CELU](#celu)
- [Softmax](#softmax)

## Sigmoid

**Location**: [`sigmoid.py`](sigmoid.py)

**Formula**: 

![](https://quicklatex.com/cache3/7a/ql_668dc0601663ef0c48396ed1ac15617a_l3.png)

*Description*: The sigmoid function confines its input within the range of 0 and 1.

**Gradient**:

![](https://quicklatex.com/cache3/1e/ql_bda49ddfef38d523f27009a45c28b61e_l3.png)

Notably, this gradient can induce the vanishing gradient issue for extremely high or low values.

## Tanh

**Location**: [`tanh.py`](tanh.py)

**Formula**:

![](https://quicklatex.com/cache3/9f/ql_72eebf5038a7f863b236caa209a7b09f_l3.png)

*Description*: The tanh function bounds its input within the range of -1 and 1.

**Gradient**:

![](https://quicklatex.com/cache3/5a/ql_1aec5bd7f300437b4a6a9f9342a6165a_l3.png)

## ReLU

**Location**: [`relu.py`](relu.py)

**Formula**: 

![](https://quicklatex.com/cache3/a0/ql_37999e1feff124baca2413a754b3bfa0_l3.png)

*Description*: ReLU imparts non-linearity in models without perturbing the receptive fields of convolutions.

**Gradient**: 

![](https://quicklatex.com/cache3/41/ql_f90d9e252e775d7a7c0591082d1fd941_l3.png)

Being computationally efficient, ReLU can occasionally cause dead neurons during the training process.

## Swish

**Location**: [`swish.py`](swish.py)

**Formula**:

![](https://quicklatex.com/cache3/8d/ql_70016b335ea865d9640e6af078f4e08d_l3.png)

Where: ![](https://quicklatex.com/cache3/e8/ql_9a315236dfcda864a869107144a3fbe8_l3.png) is a learnable parameter.

*Description*: Swish stands as a self-gated function, synthesizing the merits of both ReLU and sigmoid.

**Gradient**: 

![](https://quicklatex.com/cache3/00/ql_e0b16a0e5c70dc33fae2dd6df8f09400_l3.png)

## CELU

**Location**: [`celu.py`](celu.py)

**Formula**: 

![](https://quicklatex.com/cache3/f3/ql_c5273c8c3683571ded65e128719665f3_l3.png)

*Description*: CELU is an extension of the exponential linear units (ELU), enhanced with a scalable parameter 
![](https://quicklatex.com/cache3/a0/ql_0c3e2deb84c57937afcc3a11a786fea0_l3.png)

**Gradient**:

![](https://quicklatex.com/cache3/a9/ql_f5f8b0d44fbd0efab3c215f8bf8ea6a9_l3.png)

## Softmax

**Location**: [`softmax.py`](softmax.py)

**Formula**:

![](https://quicklatex.com/cache3/59/ql_9798671b5f273c3282d12bd273d73b59_l3.png)

*Description*: The softmax function is essential for multi-class categorization tasks, transmuting inputs into a
probability distribution spread across multiple categories.

**Gradient**: Evaluating the gradient of softmax demands intricate attention due to the inherent normalization.
This computation typically necessitates the use of the _Jacobian matrix_.
