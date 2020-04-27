---
title: "DLS Basics: Regularization"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  -
---

# setting up machine learning

## train / holdout / test

before big data: 60:20:20

now: usually enough to have 10_000 hold and test

- mismatched train/test distribution
  - make sure hold and test come from same distribution
- it might be OK to not have a test set (and only a holdout set)
  - evaluate on holdout set

## bias / variance

deep learning has less of a bias/variance tradeoff than before

back in the day, we didn't have tools that would only affect bias or variance
- usually helped one and hurt another

- high bias -- underfitting
  - poor training data performance
    - more hidden units
    - more hidden layers
    - more training epochs
- high variance -- overfitting
  - poor relative holdout set performance
    - more data (not helpful for high bias)
    - regularization
    - a more optimized NN architecture

optimal (bayes) error: what a human would get?

error as a metric for bias/variance tradeoff
- variance to mean difference between train and hold errors?
- train error << hold error
  - high variance
- train error ~ hold error, both high
  - high bias
- train error << hold error, both high
  - high bias and high variance
  - could be high bias in some regions and high variance in other regions

so long as you are regularizing, there is no cost to having a larger network (other than computation)

# proper initialization of w and b
he initialization

# regularization

https://www.youtube.com/watch?v=Q81RR3yKn30

ridge regression tries to minimize the sensitivity of our features to our target.

if you suspect your nn is overfitting data, add regularization (or get more data)

recall for logistic regression, we want to minimize the cost function $J(w, b)$:

$$J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

To add reguarlization, we can introduce a regularization parameter $\lambda$:

$$J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}||w||_2^2$$

L2 regularization

$$\frac{\lambda}{2m}||w||_2^2 = \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2 = \frac{\lambda}{2m}w^Tw$$

L1 regularization
$$\frac{\lambda}{2m}||w||_1 = \frac{\lambda}{2m}\sum_{j=1}^{n}|w_j|$$

$\lambda$ determined via holdout set.

in neural network:

$$J(w^{[1]},b^{[1]}, \cdots, w^{[L]},b^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

Where L is number of layers

frobenius norm formula:

$$||w^{[l]}||_F^2 = \sum_{i=1}^{n^l}\sum_{j=1}^{n^{l-1}}(w_{i,j}^{[l]})^2$$

because $W$ is $n^{[l]} \times n^{[l-1]}$ matrix.

dw[l] = dw from backprop + $\frac{\lambda}{m}w^{[l]}$

$W^{[l]} = W^{[l]} - \alpha dw$

L2 regularization also called "weight decay" because we are multiplying weight matrix by (1 - alphalambda/m)

## why do we care about regularization

high lambda will reduce weights of a lot of units to close to 0, which emulates a much smaller network which is less prone to overfitting

tends towards high bias

if lambda large, w small, meaning z is small because z = w[l]a[l-1]

if z is small that means it is more likely to be confined to the linear portion of sigma(z)

# dropout regularization

include graphic

introduce probability of removing hidden node which removes outgoing weights to those nodes

inverted dropout (prob that it will be kept)

dl = np.random.rand(al.shape[0], al.shape[1]) < keep_prob

al = np.multiply(al, dl)

al /= keep_prob (to keep expected value al to be the same since 0.2 values are 0)

## understanding dropout

looking at each units, it means it can't fully rely (put too much weight) on any one feature since it might drop out. this shrinks squared norm of weights. similar to L2

can also change keep_prob by weight layer

have lower keep_probs for larger weight matrices (lots of parameters) to prevent overfitting

# other regularization methods

## data augmentation
- transforming an image
- cropping an image

## early stopping

graph of error vs iterations
early stopping stops training once the dev set error goes up

orthogonalization -- thinking about one task at a time
- w, b
  - optimize cost function J
    - gradient, adam, etc.
- not overfit
  - regularization

early stopping breaks orthogonality of these two tasks
- not optimizing J to its fullest
- trying to overfit

might be better to just use L2
- might have to gridsearch across multiple values of lambda