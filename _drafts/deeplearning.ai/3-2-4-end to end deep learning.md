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

# end-to-end deep learning

when a single NN architecture can replace multiple stages of processing

speech recognition example:

- audio
- MFCC to extract features
- finding phonemes
- words
- transcript

some challenges:

- you might need a very large amount of data

how might multi-stage processing be better?

- the problem you are solving can be made much simpler with each step of processing
- there might be a large amount of data for the simpler tasks

more examples

- machine translation (very easy to map x to y, appropriate for end-to-end approach)
- assessing medical images

## when to use this approach

pros:
- lets the data speak
  - direct mapping from x to y
  - prevents algorithm from thinking with human preconceptions
- less hand-designing components

cons:
- needs a large amount of data
- excludes potentially useful hand-designed components
  - no way of injecting knowledge into the algorithm

hand-designed components tend to work more effectively with smaller datasets

the key question: do you have sufficient data to learn a sufficiently complex function to map x to y?

example: autonomous driving

- "visual" input (image, lidar, whatever)
- detect cars, pedestrians, etc.
- plan route
- execute actions (steering, acceleration, etc.)

we can use deep learning to solve image detection, but we do not have sufficient data to directly map x (visual input) to y (control of vehicle) for a true end-to-end ML approach