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

It is always needed in real world practice to find out human level performance on both train set and dev set when data mismatch occurs because both data sets are from different distributions. In such data mismatch situation, the human-level performance on data from train set is no longer representative for human-level performance on the data from dev set.

# training and testing on different distributions

cat app example

- large dataset from irrelevant source vs. small dataset from relevant source

option 1:
- put both datasets together and randomly shuffle into train dev test split
- advantage: training, dev, test sets all come from same distribution
- disadvantage: dev set will come from a different distribution than the one you actually care about

option 2:
- split train to have both sources (reserve half of smaller for dev and test total)
- have dev and test sets only from the relevant dataset
- advantage: aiming target where you want it to be
- disadvantage - training distribution different from dev and test set distributions

option 2 has better performance in the long term

# bias and variance with mismatched data distributions

suppose we have 1% training error and 10% dev error.

- if dev data came from the same distribution as training, there is a large variance problem. algorithm is not generalizing well from training set.
- if dev data comes from different distriution, we cannot safely make this conclusion. eg maybe it's doing just fine on the dev set, but the dev set is harder to make predictions on

workaround: the training-dev set
- just as dev and test sets will have the same distribution, the training set and training-dev set will have the same distribution
- look at the error on both the train-dev and dev sets

two scenarios:

- variance problem: 1% training error, 9% train-dev error, and 10% dev error. train-dev comes from the same distribution as train.
- data mismatch problem: 1% training error, 1.5% train-dev error, and 10% dev error. data mismatch problem, since dev comes from different distribution than train.

you get an idea of the issues in performance:
- human level error
- training set error
- training-dev error
- dev set error
- test error

gives rise to:
- avoidable bias
- variance
- data mismatch
- degree of overfitting to the dev set

What about this?
| error        | Are           | Cool  |
| ------------- |-------------| -----|
| human      | right-aligned | $1600 |
| examples trained on      | centered      |   $12 |
| examples not trained on | are neat      |    $1 |

# addressing data mismatch
- carry out manual error analysis to try to understand difference between training and dev/test sets
  - eg: analyze examples to figure out how your dev set is different than your train set
- when you have insight into the nature of your dev set errors or insight into how the dev set is different or more difficult than your training set, try to find ways to make your training data more similar
  - artificial data synthesis (eg clean audio + car noise)
  - if you have 10000 hours of clean audio and only 1 hour of car noise, you may be tempted to loop the 1 hour 10000 times, but the algorithm may overfit to this small subset of "car noise" space