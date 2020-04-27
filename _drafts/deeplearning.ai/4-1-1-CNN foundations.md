---
title: "DLS: Convolutional Neural Networks"
excerpt_separator: "<!--more-->"
categories:
  - python
  - data science
tags:
  - deeplearning.ai
  - neural networks
  - convolutional neural networks
---

# The Problem with Computer Vision

include pictures for each
- image classification
- object detection
- neural style transfer

one of the challenges of computer vision problems is that the inputs can get quite big.

example: processing a 1 megapixel RGB image

$\mathbf{x} \in \mathbb{R}^{3M}$

if we have a 1000-neuron hidden layer, then W is a $1000, 3M$ dimensional matrix with 3 billion elements.

# Convolutions and Edge Detection

how to detect edges?

example with 6x6 matrix and convolve with 3x3 filter (sometimes called [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)))

(in mathematics, the asterisk is the standard operator for convolution)

conv-forward
tf.nn.conv2d
keras.conv2d

show example with 6x6 and a bunch of 10s

more filters:
- sobel filter
- scharr filter

deep learning can learn whatever filter it needs to detect edges

# padding

take 6x6 matrix and convolve with 3x3 filter you get 4x4 matrix

nxn matrix * fxf filter = (n-f+1) result

to solve shrinking output and throwing away info from edges / imbalanced weights

we can pad the image with an additional border. by convention we pad with 0s by a border of 1

"valid" : no padding
"same" : pad so output size is the same as input size

p = (f-1) / 2

by convention, f is almost always odd
- padding formula
- filter has a center

# strided convolutions

instead of moving by 1, move by stride $s$.

output is (n + 2p - f) / s + 1

if fraction is not integer, we take floor

# a note about convolution vs. cross-correlation

we've skipped mirroring operation and we're actually doing cross-correlation. in deep learning literature we just call it cross-correlation.



