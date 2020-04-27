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

# transfer learning

you can take an algorithm on one task and maybe use part of that knowledge to help you do a better job performing another task

picture showing this

pretraining -> fine-tuning

take existing NN architecture and:
- remove final output layer and corresponding weights
- initialize random weights and map to new output, retrain NN

the reason this works: a lot of low-level features (edge detection, detecting curves, etc.) could help in another context

you could even add several new layers to the neural network

## when does transfer learning make sense?

- when you have a lot of data you're transferring from and relatively few data for the new task
- would NOT make sense if size of dataset is similar

# multi-task learning

computer vision

you have one NN attempt to do several things at the same time. each task then (hopefully) helps all other tasks

let's start with an example where we have $t$ tasks to model:

$$\mathbf{y}^{(i)} =
\begin{cases}
\text{pedestrians} & \quad0 \\
\text{cars} & \quad 1 \\
\text{stop signs} & \quad 1\\
\text{traffic lights} & \quad 0\\
\vdots & \quad \vdots
\end{cases}
$$

then:

$$Y = \begin{bmatrix}
| & | & & |\\
\mathbf{y}^{(1)} & \mathbf{y}^{(2)} & \cdots & \mathbf{y}^{(m)}\\
| & | & & |
\end{bmatrix}$$

where $Y$ is a $t \times m$ dimensional matrix

and:

$$\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{t}\mathcal{L}(\hat{y}_{j}^{(i)}, y_{j}^{(i)})$$

sum only if value at j is 0 or 1 (allows for unlabeled tasks)

unlike softmax regression, each example can have multiple labels.

## when does multi-task learning make sense?

if all three conditions are satisfied:
- training on a set of tasks that could benefit from having shared lower-level features
- the amount of data you have for each task is similar
- train a NN that is big enough to do well on all tasks

multi-task learning used less frequently than transfer learning