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

# multi-class classification

softmax is a generalization of logistic regression to more than two classes

C classes

indexed: (0, 1, ..., C - 1)

$n^{[L]} = C$

$$\mathbf{z^{[L]}} = W^{[L]}\mathbf{a^{[L-1]}} + \mathbf{b^{[L]}}$$

$$t_{i} = e^{z^{[L]}_i}$$

$$\mathbf{\hat{y}} = \mathbf{a^{[L]}} = g^{[L]}(\mathbf{z^{[L]}}) = \frac{\mathbf{t}}{\sum_{i=1}^{C}t_{i}} = \begin{bmatrix}
P(1) \\ P(2) \\ \vdots \\ P(C)
\end{bmatrix}$$

maybe show example using iris dataset?

## why softmax?

hardmax like [1, 0, 0, 0]

if C=2, softmax reduces to logistic regression

## loss function

$$\mathcal{L}(\mathbf{\hat{y}}, \mathbf{y}) = -\sum_{j=1}^{C}y_j\log(\hat{y}_j)$$
$$\mathcal{J}(W^{[1]}, \mathbf{b}^{[1]}, \cdots, W^{[L]}, \mathbf{b}^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\mathbf{\hat{y}}, \mathbf{y})