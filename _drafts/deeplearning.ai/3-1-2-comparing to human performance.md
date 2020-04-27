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

# comparing to human-level performance

after surpassing human-level performance, accuracy progress slows down and asymptotically approaches bayes optimal error (best possible error)

humans are quite good at a lot of tasks, so maybe there is not that much headroom

if it's worse, you can:
- get labeled data from humans
- manual error analysis (why did a person get it right)
- better analysis of bias/variance

# avoidable bias

comparing to human-level performance allows you to better evaluate your algorithm

if large discrepancy between human vs. training, focus on bias.

if very small discrepancy between human vs. training, maybe focus on variance

human-level error as a proxy for bayes error for tasks humans are good at

difference between human-level error and training error "avoidable bias"

difference between training error and dev error "variance"

# human-level performance, defined

depends on how you wish to deploy your algorithm.

be clear what your purpose is.

if it's to show you can surpass a single human, perhaps typical performance.

if it's to show human error as a proxy for bayes error, then team of doctors is better.

- human-level error (proxy for bayes error)
  - avoidable bias
- training error
  - variance
- dev error

there is no particular expectation that you should get 0% error

# surpassing human-level performance

- team of humans 0.5
- one human 1.0
- training error 0.6
- dev error 0.8

what is avoidable bias?

- team of humans 0.5
- one human 1.0
- training error 0.3
- dev error 0.4

is avoidable bias 0.3? or have we overfit to our training data?

# situations where ML siginificantly surpasses human-level performance

- online advertising
- product recommendations
- logistics
- loan approvals

all four examples draw from huge amounts of structured data (not natural perception problems)

humans tend to be better at natural perception tasks

ML approaching human-level performance:
- speech recognition
- image recognition
- medical diagnoses

# putting it all together: improving model performance

the two fundamental assumptions of supervised learning:

- you can fit the training set well
- training set performance generalizes pretty well to the dev/test set

if you want to improve the performance of your ML algorithm:

- look at avoidable bias
  - train a bigger model
  - train for longer
  - use better optimization algorithms
    - momentum
    - RMSprop
    - adam
  - change NN architecture
    - NN, RNN, or CNN
    - gridsearch over hyperparameters
- look at variance
  - more data
  - regularization
    - L2, dropout, data augmentation
  - change NN architecture