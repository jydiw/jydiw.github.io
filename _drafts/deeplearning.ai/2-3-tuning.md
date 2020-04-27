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

# hyperparameters

- alpha

- beta (if you're using momentum)
- hidden units
- mini batch size


- layers
- learning rate decay

- beta1, beta2, epsilon (if you're using adam)

## how to choose hyperparameters
- try randomized values.
  - don't use a grid! sample within the space.
- go coarse to fine
  - still take random samples

## choosing scale of hyperparameters
- do not choose uniformly random over scale of valid values
  - maybe OK for number of hidden layers or number of hidden nodes
- choose over log scale uniformly at random

```python
# alpha
# if you want to sample between 0.0001 and 1
# 10 ** r where r between -4 and 0
r = np.random.uniform(-4, 0)
alpha = 10 ** r

# beta
# if you want to sample between 0.9 and 0.999
s = np.random.uniform(-3, -1)
beta = 1 - 10 ** s
```

## re-evaluate your hyperparameters

two schools of thought:

1. panda: babysit one model
    - huge dataset but not a lot of computational resources
    - change one parameter at a time, one day at a time
2. caviar: train many models in parallel
    - analyze multiple learning curves at once

# batch normalization

when training a model like logistic regression, normalizing can speed up learning by optimizing gradient descent.

in deeper models, not only do we have input layer X but activation layers  we can also normalize Z to train W and b faster (there is debate whether to normalize A or Z).

given values $z[l](1) ... z[l](m)$

compute mean, compute variance, normalize (divide by sd + epsilon)

Input: $\mathbf{x}$ over minimatch $\mathcal{B}$. parameters to learn: $\gamma$, $\beta$.
Output: $\{y_i = BN_{\gamma, \beta}(x_i)\}$

$$\mathbf{z}^{\{\mathcal{B}\}[l]} = W^{[l]}\mathbf{a}^{[l]}$$

$$\mu^{\{\mathcal{B}\}[l]} = \frac{1}{m}\sum_{i}^{m}z^{\{\mathcal{B}\}[l](i)}$$

$$(\sigma^{\{\mathcal{B}\}[l]})^2 = \frac{1}{m}\sum_{i}^{m}(z^{\{\mathcal{B}\}[l](i)} - \mu^{[l]}_{\mathcal{B}})^2$$

$$z_{norm}^{\{\mathcal{B}\}[l](i)} =
\frac{z^{[l](i)}
- \mu^{[l]}}{\sqrt{\sigma^{[l]2}_{\mathcal{B}}
- \epsilon}}$$

$$\tilde{z}^{[l](i)} = \gamma^{[l]} z_{norm}^{[l](i)} + \beta^{[l]} \equiv BN_{\gamma, \beta}(x_i)$$
>notice that if $\gamma^{[l]} = \sqrt{\sigma^{[l]2}_{\mathcal{B}} - \epsilon}$ and $\beta^{[l]} = \mu^{[l]}_{\mathcal{B}}$, it would exactly invert the normalization process.
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{\tilde{z}}^{[l]})$$

## how to fit batch normalization into a deep neural network

show picture of how batch normalization works
- normal NN: each neuron split into z[l] -> a[l]
- BN NN: x -> z -> z~ -> g(z~) -> a

parameters
- normal NN: W, b ...
- batchnorm: beta, gamma
  - dbeta: beta = beta - alpha dbeta
  - because we are normalizing everything, no need for bias term

```python
for t in range(1,num_minibatch + 1):
    # computer forward prop
        # use BN function to form ztilde
    # backprop to compute dW, dbeta, dgamma
    # update parameters
```

## why batch normalization works

### covariate shift

shallow network but trained on only black cats

test data has color cats, might not do that well

"covariate shift"

let's look via perspective of third layer

W[3], b[3] gets some values a[2] and needs to find a way to map them to yhat.

since normalizing input layer x helps with gradient descent, makes sense that normalizing z helps as well.

prevents covariate shift from layer to layer

### regularization
- each **mini-batch** is scaled by the mean/variance of that mini-batch
- this adds some noize to the values z[l], so it adds noise to each hidden layer's activations
- this has a *slight* regularization effect

# batch normalization at test time

use exponentially weighted average across different mini-batches

weighted average is used at test time

$$\mu^{\{1\}[l]}$$