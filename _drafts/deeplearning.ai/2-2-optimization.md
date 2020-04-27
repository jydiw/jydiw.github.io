---
title: "DLS Basics: Optimization Algorithms"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  -
---
# optimizing your neural network

## normalizing inputs
- subtract mean
- normalize mean

$$\mu = \frac{1}{m} \sum_{i=1}^{m}x^{(i)}$$
$$x \coloneqq x - \mu$$

$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m}(x^{(i)})^{\circ2}$$

normalizing will help with gradient descent

## vanishing / exploding gradients
wtf?

## weight initialization for deep networks
$$z =  \sum_{i=1}^{n} w_ix_i + b$$
in order for z to not be very large, you want wi to be smaller for larger n

ideally you want variance(wi) = 1/n (2/n for ReLU)
- tanh is 1 (xavier initialization)

W = np.random.randn(shape) * sp.sqrt(1/n^[l-i])

# numerical approximation of gradients

formal definition of symmetric derivative:

$$f^{\prime}(\theta) = \lim_{\epsilon \to 0}\, \frac{f(\theta + \epsilon) - f(\theta-\epsilon)}{2\epsilon}$$

for nonzero $\epsilon$, the derivative can then be approximated as:

$$f^{\prime}(\theta) \approx \frac{f(\theta + \epsilon) - f(\theta-\epsilon)}{2\epsilon}$$

where the error is $O(\epsilon^2)$

this differs from the difference quotient we are typically used to in calculus

$$f^{\prime}(\theta) = \lim_{\epsilon \to 0}\, \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$$

Where the error is $O(\epsilon)$ for nonzero $\epsilon$. For small values of $\epsilon$, this provides a much more accurate approximation for the derivative.

## gradient checking

take W1, b1, ... WL, bL and reshape/concatenate into vector theta. same with dW, db etc.

for each value i:

$$\frac{\partial J}{\partial \theta_{approx\,i}} \coloneqq \frac{J(\theta_1, \theta_2, \cdots, \theta_i + \epsilon, \cdots) - J(\theta_1, \theta_2, \cdots, \theta_i - \epsilon, \cdots)}{2\epsilon}$$

- don't use in training. only to debug
- if algorithm fails grad check, look at components to identify bug
- remember regularization
- doesn't work with dropout
  - check that it works without dropout and then implement it


# batch vs mini-batch gradient descent

recall vectorization to compute $m = 5000000$ samples.

batch is taking all of your training examples

let's say we make mini-batches of 1_000 each.

$X^{\{ 1\}}$ is first mini-batch

- $X^{(i)}$ is the $i$th example
- $X^{[L]}$ is the $L$th layer
- $X^{\{\mathcal{B}\}}$ is the $\mathcal{B}$th mini-batch

```python
for t = 1, ..., 5000:
    forward prop on X{t}
        Z[1] = W[1]X{t} + b[1]
        A[1] = g[1](Z[1])
        ...
        A[L] = g[L](Z[L])
    J{t} = 1/1000 sum L(yhat, y) + L2
    backprop using X{t}, Y{t}
    W[l] := W[l] - alpha dW[l]
    b[l] := b[l] - alpha db[l]
```
this is one "epoch"

1. stochastic gradient descent
   1. no benefit from vectorization
2. in-between
   1. fastest learning
   2. makes progress without using entire training set
   3. typical mini-batch sizes: 64-512
   4. make sure that a single minibatch actually fits in the CPU/GPU memory
3. batch gradient descent
   1. could take too long for each iteration
   2. OK if m <= 2000

# exponentially weighted moving averages

$$v_t = \beta v_{t-1} + (1-\beta) \theta_t \approx \frac{1}{1-\beta}\theta_t$$

$$\beta = 0.9 \quad \approx \text{10 days' MA}$$
$$\beta = 0.98 \quad \approx \text{50 days' MA}$$

$\beta$ is a hyperparameter

$$v_t = (1-\beta) \theta_t + \beta(1-\beta)\theta_{t-1} + \cdots + \beta^{(t-1)}(1-\beta)\theta_1$$

```python
v = 0
v = beta * v + (1 - beta) * theta_1
```

not as accurate as actual moving average, but less computationally intensive

## bias correction
above calculation has a lag since v0 = 0

$$\frac{v_t}{1-\beta^t}$$

# gradient descent with momentum
compute exponentially weighted moving average of your gradients, then use THAT gradient to update weights

on iteration t:
compute: dw, db on minibatch

$$v_{dW} = \beta v_{dW} + (1-\beta) dW$$

vdw = betavdw + (1-beta)dw

beta usually 0.9

# RMS prop

on iteration t:
  computer dW, db on current mini-batch
  $$S_{dW} = \beta S_{dW} + (1 - \beta) dW^{\circ   2}$$

  $$W \coloneqq W - \alpha \frac{dW}{\sqrt{S_{dW}}}$$
  $$b \coloneqq b - \alpha \frac{db}{\sqrt{S_{db}}}$$

  dW large
  db small

# adam (GD+M and RMSprop)

**Ada**ptive **M**oment Estimation

hyperparameters:

- alpha: needs to be tuned
- beta: 0.9 (MWA of dW)
- beta2: 0.999 (MWA of dWcirc2)
- epsilon: 1E-8

# Learning Rate Decay

slowly reducing the value of alpha as learning approaches convergence

1 epoch = 1 pass through the data

$$\alpha = \frac{1}{1 + decayrate \cdot epoch}$$

$$\alpha = 0.95^{epoch} \cdot \alpha_0$$

$$\alpha = \frac{k}{\sqrt{epoch}} \cdot \alpha_0$$

# The problem of local minima

- actually very unlikely to get stuck in bad local optima
- most points with gradient = 0 are saddle points, not local minima
- we actually have to worry about plateaus, where gradient is close to 0 but not quite
- plateaus make learning very slow