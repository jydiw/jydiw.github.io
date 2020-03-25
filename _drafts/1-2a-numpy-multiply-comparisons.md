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

# A note about different `numpy` array methods

Most `numpy` methods are element-wise.

## 1D and scalar

```python
a = 4
b = np.array([1, 3, 5])

print(np.multiply(a, b))  # broadcasts scalar across array
print(np.dot(a, b))       # equivalent to multiply since a is scalar
print(np.matmul(a, b))    # throws an error since a is scalar
```

```
[ 4 12 20]
[ 4 12 20]
ValueError: ...
```

## 1D and 1D

```python
a = np.array([2, 4, 6])
b = np.array([1, 3, 5])

print(np.multiply(a, b))  # element-wise multiplication
print(np.dot(a, b))       # dot product of two vectors
print(np.matmul(a, b))    # equivalent to dot() since both 1D
```

```
[ 2 12 30]
44
44
```

## 1D and 2D

```python
a = np.array([2, 4, 6])
b = np.array([[1, 3, 5],
              [7, 9, 11],
              [13, 15, 17]])

print(np.multiply(a, b))  # element-wise multiplication
print(np.dot(a, b))       # same as matmul() since a is 1D
print(np.matmul(a, b))    # elevates a to 1xd matrix
```

```
[[  2  12  30]
 [ 14  36  66]
 [ 26  60 102]]
[108 132 156]
[108 132 156]
```

```python
print(np.matmul(b, a))    # returns np.dot(b1, a), ..., np.dot(b3, a)
```

```
[ 44 116 188]
```