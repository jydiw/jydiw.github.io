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

# Vectorized Regressions

Recall the general form of the linear regression equation with $n$ features:

$$z =  \sum_{i=1}^{n} w_ix_i + b$$

> From this point forward, we will be swapping out statistical language with terms more commonly used in the machine learning community:
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

> We'll find that expressing this product as the matrix multiplication of the transpose allows us to scale up this notation to vectorize neural networks.

It follows that the linear regression equation can be written as:

$$z = \mathbf{w}^T \mathbf{x} + b$$

## Vectorized Regression with $m$ Examples

We could then represent multiple examples with unique vectors where $\mathbf{x}^{(j)}$ is the $j$th example vector with feature values $x_1^{(j)}$, $x_2^{(j)}$, etc.:

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

From the previous section, $z = \sum_{i=1}^{n} w_ix_i + b$. It follows that:

$$\mathbf{z} = \begin{bmatrix}
z^{(1)} & z^{(2)} & \cdots & z^{(m)}
\end{bmatrix} = \begin{bmatrix}
\sum_{i=1}^{n} w_ix_i^{(1)} + b & \sum_{i=1}^{n} w_ix_i^{(2)} + b & \cdots & \sum_{i=1}^{n} w_ix_i^{(m)} + b
\end{bmatrix} = \mathbf{w}^{\mathrm{T}}X + b$$

>Those who are mathematicians may balk at the above notation for $b$.  not changing with each example, $\mathbf{b}$ has to be the same dimension as $\mathbf{w}^{\mathrm{T}}X$ (that is, a $1 \times m$ matrix) in order for the matrix addition to work.

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
[249732.27694854]
for loop: 340.1 ms
```

Using vectors, we would write the following:

```python
start = time.time()
z = np.dot(w, x) + b
end = time.time()

print(z)
print('dot product: ' + str(1000*(end-start))[:5] + ' ms')
```
```
[249732.27694854]
dot product: 0.966 ms
```
This is over a 300x speed improvement for the same calculation!