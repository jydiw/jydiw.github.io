---
title: "DLS Basics: Vectorization and Broadcasting"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  -
---

Vectorization is the process of expressing our calculations as matrix operations. This allows us to have concise notation as well as utilize linear algebra packages (such as `Numpy`) to speed up our work.

Broadcasting describes the process of applying different arithmetic operations to arrays of different shapes.

# Vectorized Regressions

Recall the general form of the linear regression equation with $n$ input variables:

$$\hat{y} = \hat{\beta_0} + \sum_{i=1}^{n} \hat{\beta}_ix_i$$

We will rewrite the above expression to align with what is followed in the machine learning community:

$$z = b + \sum_{i=1}^{n} w_ix_i$$

> From this point forward, we will be removing the "hats" from our symbols and swapping out statistical language with terms more commonly used in the machine learning community:
>
>- $x_i$ -- the $i$th *feature*
>- $w_i$ -- the $i$th *weight*
>- $b$ -- the *bias*

Suppose we have column vectors $\mathbf{w}$ and $\mathbf{x}$ to represent our $n$ weights and features, respectively:

$$\mathbf{w} = \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n \\
\end{bmatrix} \qquad \mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{bmatrix}
$$

If we multiply $\mathbf{w}^T$ (a $1 \times n$ matrix) with $\mathbf{x}$ (an $n \times 1$ matrix), we get the matrix multiplication equivalent of the dot product $\mathbf{w} \cdot \mathbf{x}$, which is equal to $\sum_{i=1}^{n} w_ix_i$:

$$\mathbf{w}^T \mathbf{x} = \begin{bmatrix}
w_1 & w_2 & \cdots & w_n
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{bmatrix} = w_1x_1 + w_2x_2 + \cdots + w_nx_n = \sum_{i=1}^{n} w_ix_i$$

> Note: Using matrix multiplication allows us to scale up this notation.

It follows that the *vectorized* linear regression equation can be written as:

$$z = \mathbf{w}^T \mathbf{x} + b$$

## Vectorized Regression with $m$ Examples

Vectorizing our regression allows us to quickly represent how we would apply regression to multiple examples. We could then represent multiple examples with unique vectors, where the $j$th example vector is written as a superscript with round brackets: $\mathbf{x}^{(j)}$:

$$\mathbf{x}^{(j)} = \begin{bmatrix}
|\\
\mathbf{x}^{(j)}\\
|
\end{bmatrix} = \begin{bmatrix}
x_1^{(j)} \\
x_2^{(j)} \\
\vdots \\
x_n^{(j)} \\
\end{bmatrix}
$$

If we stack $m$ $n$-dimensional feature vectors horizontally, we form an $n \times m$ matrix $X$:

$$X = \begin{bmatrix}
| & | & & |\\
\mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)}\\
| & | & & |
\end{bmatrix} = \begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)}\\
x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(m)} \\
\vdots & \vdots & \ddots & \vdots \\
x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)} \\
\end{bmatrix}
$$

such that row vector $\mathbf{w}^{\mathrm{T}}$ can multiply with each column vector $\mathbf{x}^{(j)}$ to get $\sum_{i=1}^{n} w_ix_i^{(j)}$:

$$\mathbf{w}^{\mathrm{T}}X =  \begin{bmatrix}
w_1 & w_2 & \cdots & w_n
\end{bmatrix}\begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)}\\
x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(m)} \\
\vdots & \vdots & \ddots & \vdots \\
x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)} \\
\end{bmatrix} = \begin{bmatrix}
\sum_{i=1}^{n} w_ix_i^{(1)} & \sum_{i=1}^{n} w_ix_i^{(2)} & \cdots & \sum_{i=1}^{n} w_ix_i^{(m)}
\end{bmatrix}$$

## Broadcasting our bias $b$
From the previous section, $z = \sum_{i=1}^{n} w_ix_i + b$. It follows that:

$$\mathbf{z} = \begin{bmatrix}
z^{(1)} & z^{(2)} & \cdots & z^{(m)}
\end{bmatrix} = \begin{bmatrix}
\sum_{i=1}^{n} w_ix_i^{(1)} + b & \sum_{i=1}^{n} w_ix_i^{(2)} + b & \cdots & \sum_{i=1}^{n} w_ix_i^{(m)} + b
\end{bmatrix} = \mathbf{w}^{\mathrm{T}}X + b$$

>$\mathbf{b}$ has to be the same dimension as $\mathbf{w}^{\mathrm{T}}X$ (that is, a $1 \times m$ matrix) in order for the matrix addition to work. Here we *broadcast* the operation of adding $b$ over the entire array of $\mathbf{w}^{\mathrm{T}}X$--that is, we add $b$ to every value in the array.

We can then represent the predictions as:
$$\mathbf{\hat{y}} = g(\mathbf{z}) = \begin{bmatrix}
g(z^{(1)}) & g(z^{(2)}) & \cdots & g(z^{(m)})
\end{bmatrix}$$

Putting it all together, the vector representation for a logistic regression with $n$ features and $m$ examples is:

$$\mathbf{z} = \mathbf{w}^{\mathrm{T}}X + \mathbf{b}$$
$$\mathbf{\hat{y}} = g(\mathbf{z})$$

# Vectorization and Broadcasting in `numpy`

Vectorizing our code allows for faster calculations. Machine learning often deals with large datasets, so we always want to compute as efficiently as possible.

Let's codify $z =  \sum_{i=1}^{n} w_ix_i + b$ with a million features:

```python
import time
import numpy as np

np.random.seed(1)

n = 1_000_000
w = np.random.rand(n)
x = np.random.rand(n)
b = np.random.rand(1)
```
We may be tempted to try this:
```python
z = 0
start = time.time()
for i in range(n):
    z += w[i]*x[i]
z += b
end = time.time()

print(z)
print('for loop: ' + str(1000*(end-start))[:5] + ' ms')
```

```
[249825.0610044]
for loop: 330.1 ms
```

Using vectors, we would write the following:

```python
start = time.time()
# note that np.dot(w, x) and b have different shapes
z = np.dot(w, x) + b
end = time.time()

print(z)
print('dot product: ' + str(1000*(end-start))[:5] + ' ms')
```
```
[249825.06100441]
dot product: 1.001 ms
```
This is over a 300x speed improvement for the same calculation!