---
title: "DLS: CNN Case Studies"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

turns out that CNNs trained well in one computer vision task often work well on other tasks.

classic networks:
- LeNet-5
- AlexNet
- VGG-16

Other networks:
- ResNet
- Inception

# LeNet-5

The goal of LeNet-5 was to recognize handwritten digits and was trained on grayscale images.

more modern implementation:
- would probably use relu instead of sigmoid/tanh
- typically keep number of channels the same for filters

# AlexNet
Lots of similarities to LeNet-5, but:
- much more parameters (around 60M)
- uses relu instead, so much faster
- modern implementation won't use local response normalization

# VGG-16
(16 weight layers)

uses a much simpler network: CONV = 3x3 s=1 same
and all maxpool = 2x2, s=2

138M parameters

# ResNet

The residual block

consider two layers in a neural network where you start with a[l] and feed to get a[l+1], then a[l+2].

$$
\begin{array}{rcl}
z^{[l+1]}&\!\!=&\!\!W^{[l+1]} \cdot a^{[l]} + b^{[l+1]}\\
a^{[l+1]}&\!\!=&\!\!g(z^{[l+1]})\\
\\
z^{[l+2]}&\!\!=&\!\!W^{[l+2]} \cdot a^{[l+1]} + b^{[l+2]}\\
a^{[l+2]}&\!\!=&\!\!g(z^{[l+2]})
\end{array}
$$

What if we could just take a shortcut (also called skip connection)? (The shortcut arrives before the non-linearity)

$$
\begin{array}{rcl}
a^{[l+2]}&\!\!=&\!\!g(z^{[l+2]}+ a^{[l]})
\end{array}
$$

Turns out "plain" NNs will have an increase in training error with many many layers. WIth ResNet, training error should always go down, even with over 100 layers.

Helps with vanishing/exploding gradient problems.

# Why ResNets Work

$$
\begin{array}{rcl}
a^{[l+2]}&\!\!=&\!\!g(z^{[l+2]}+ a^{[l]})\\
&\!\!=&\!\!g(W^{[l+2]} \cdot a^{[l+1]}+ b^{[l+2]} + a^{[l]})
\end{array}
$$

# 1x1 convolutions

Also called "Network in Network"

Pooling layers are useful to shrink nh and nw, but 1x1 convolutions are useful to shrink nc or adding another layer of nonlinearity.

Useful if you want to change the number of channels in your volume without changing the other dimensions.

a convolution with a 1x1 filter doesn't seem particularly useful, since it seems like it's just multiplying an input by a scalar.

However, if we have multiple channels, then a 1x1 filter will take the element-wise product between the number of channels, and then apply relu nonlinearity.

A 1x1 filter acts like a fully-connected network in each of the positions in the image. It is like a single neuron accepting a single slice containing nc inputs, then multiplying them by nc weights and applying relu. Multiple units then behave like multiple neurons in a MLP.

# Inception Network

instead of needing to pick parameters, use them all and then concatenate them, allow the model to pick and choose which works best. sort of like an ensemble method.

problem of computational cost

instead you can use 1x1 convolution to reduce volume first, then use normal convolution. this is sometimes called the "bottleneck" layer.

you can chain "inception blocks" together

side branches takes a hidden layer to try to predict the output

# practical advice for using convnets

## using open source implementations

search through github for previous implementations

`git clone {url}`
`cd {dir}`
`dir`

