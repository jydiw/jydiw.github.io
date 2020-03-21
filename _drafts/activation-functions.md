---
title: "Neural Networks: Propagation, Activation Functions"
excerpt_separator: "<!--more-->"
categories:
  - tutorials
  - python
  - DLS
tags:
  - neural networks
  - activation functions
---

another article: matrix dimensions?

Z[l] = W[l]A[l-1] + b[l]

for 1 training example:
if dim Z[l] = (n[l], 1), b[l] must also be (n[l], 1)
and A[l-1] = (n[l-1], 1)
then W[l] = (n[l], n[l-1])

for m training examples:
if dim Z[l] = (n[l], m) but b[l] must also be (n[l], 1) (broadcasted)
and A[l-1] = (n[l-1], m)
then W[l] = (n[l], n[l-1])

discuss what happens to single training example then vectorize

A, X, Z = column vectors stacked horizontally
W = w.T row vectors stacked vertically

Z = outputs
A = activations

Z[l] = W[l]A[l-1] + b[l]
A[l] = g(Z[l])

logloss(A,Y)     L(a,y)      -yloga - (1-y)log(1-a)      # curly L
gprime(z)        g'(z)       da/dz = d/dz g(z)
da               dL/da = d/da L(a,y) = -y/a + (1-y)/(1-a)
dz               dL/dz = dL/da * da/dz = da * gprime(z)

random initialization
W[1] = np.random.randn((num_nodes_2, num_nodes_1)) * 0.01
# small number works better for activation function
b[1] = np.zero((num_nodes_2, 1))
W[2] = ...
b[2] = ...



Forward Propagation
Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

Cost
cost = -1 * (np.dot(Y, np.log(A2.T)) + np.dot(1-Y, np.log(1-A2.T))) / m
cost = float(np.squeeze(cost))

Back Propagation:
dZ2 = da[l] * g[l]'(z[l])
dW2 = np.dot(dZ2, A1.T) / m
db2 = np.sum(dZ2, axis=1, keepdims=True) / m
dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
dW1 = np.dot(dZ1, X.T) / m
db1 = np.sum(dZ1, axis=1, keepdims=True) / m


Why Non-Linear Activation Functions?
identity activation functions result in a linear function
ie neural network no more expressive than standard logistic regression


Types of activation functions g(z) and their derivatives

- sigmoid
  - sig(z) = 1 / (1 + e^-z)
  - sig'(z) = g(z) * (1 - g(z))
  - never use except for output layer
- hyperbolic tangent (tanh)
  - tanh(z) = (e^z - e^-z) / (e^z + e^-z)
  - tanh'(z) = 1 - (tanh(z))^2
  - almost always better than sigmoid
  - should not be used on the output layer

- if z very large or small, dg/dz close to 0

- rectified linear unit (ReLU)
  - ReLU(z) = max(0, z)
  - ReLU'(z) = 0 if z < 0; 1 if z > 0; undef if z = 0
  - less computationally intensive
  - learning is much faster
- leaky ReLU
  - LReLU(z) = max(0.01z, z)
  - LReLU'(z) = 0.01 if z < 0; 1 if z > 0; undef if z = 0
  - in practice, enough neurons will have nonzero dg/dz that learning is still possible with ReLU


what about machine learning regression?
linear function would be appropriate for output layer, but all other layers should be tanh or relu

# gradient descent
include photo of simple NN with one hidden layer
parameters: W[1], b[1], W[2], b[2]
cost function: J(W[1], b[1], W[2]. b[2]) = 1/m SUM log loss

