---
title: "Neural Networks: Deep Neural Networks"
excerpt_separator: "<!--more-->"
categories:
  - DLS
tags:
  - neural networks
  - activation functions
---

# Required Background

- some basic calculus and linear algebra
- logistic regression
- vectorization and broadcasting
- cost functions and gradient descent

# What is a neural network?



neural networks basically stacks of interconnected logistic regression objects. this allows for the network to model nonlinearities.

## Components of a neural network
- $a^{[0]} = x$ -- the input layer
-

make a picture:
![text](https://miro.medium.com/max/349/1*Pi12IKOO14pGMvh4nXbsJQ.png)



$$W = \begin{bmatrix}
\cdots & \mathbf{w}_1^T & \cdots \\
\cdots & \mathbf{w}_2^T & \cdots \\
& \vdots & \\
\cdots & \mathbf{w}_n^T & \cdots \\
\end{bmatrix}$$

$$\mathbf{Z} = W \cdot \mathbf{x} + \mathbf{b}$$

$$\mathbf{A} = g(Z) = \begin{bmatrix}
g(z_1) \\
g(z_2) \\
\vdots \\
g(z_n)
\end{bmatrix}$$

another article: matrix dimensions?

why deep representations?
CNN example of edges
            of audio clips

circuit theory?
XOR O(logn) vs O(2^n)

# initialize_parameters_deep(layer_dims):
layer dims -- array containing dimensions of each network
out -- parameters with 'W1', 'b1', ..., 'WL', 'bL'

# linear_activation_forward(A_prev, W, b, activation)
Z, linear_cache = linear_forward(A_prev, W, b)
A, activation_cache = g(Z)
return A, cache

# L_model_forward(X, parameters)

