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

# deep learning frameworks

- caffe
- CNTK
- DL4J
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- TensorFlow
- Theano
- Torch

choosing a deep learning framework
- ease of programming
- running speed
- truly open-source (with good governance)

# TensorFlow

```python
import numpy as np
import tensorflow as tf

coefficients = np.array([[1.], [-10.], [25.]])
w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3,1])
# cost = tf.add(tf.add(w**2, tf.multply(-10., w)), 25)
# cost = w**2 - 10*w + 25     # works since w is tf.Variable()
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()

session.run(init)
print(session.run(w))

session.run(train, feed_dict={x:coefficients})
print(session.run(w))
```

```
0.0
0.1
```

running the operation only puts in the computation graph but does not run the computation. you have to run the Session() in order to actually get the operation. When you specify the operations needed for a computation, you are telling TensorFlow how to construct a computation graph. The computation graph can have some placeholders whose values you will specify only later. Finally, when you run the session, you are telling TensorFlow to execute the computation graph.