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

# ML strategy

summary:
- start with a basic model to see what mistakes it makes
- compare to human performance and apply error analysis to plan how to improve algorithm performance
  - sample from mislabeled dev set
- consider adding outside data if your current dataset is small
  - make sure your dev and test set come from distributions you actually care about
- consider synthesizing data to emulate misclassified examples
  - be careful of overfitting
- make sure dev and test set belong to the same distribution
  - DL algorithms are usually robust to having different train and dev distributions
- consider transfer learning
  - only if new learning task is similar and has less data
- consider multi-task learning
- if you have an enormous amount of data that could easily map from X to Y, consider an end-to-end approach

what to do to increase accuracy:

- collect more training data
- get a more diverse training set (both positive and negative)
- train longer
- try a different optimization algorithm
- change the size of the network
- try any of the various regularization methods

if you choose poorly, you might realize some strategies lead you astray

# orthogonalization

make controls that do just one thing

make each control relatively interpretable

1. first make sure that algorithm performs well on training set. if not:
   1. bigger network
   2. optimization algorithm
2. dev set
   1. regularization
   2. bigger training set
3. test set
   1. bigger dev set
4. real world
   1. change dev set
   2. cost function


non-orthogonal methods:
- early stopping
  - affects both training and dev set

# single number evaluation metric

idea-code-experiment loop

precision:
- accuracy of positive predictions
- how many selected items are relevant?

$$\text{precision} = \frac{\text{true positive}}{\text{true positive + false positive}}$$

recall / sensitivity:
- probability of detection
- what proportion of relevant items are selected?

$$\text{recall} = \frac{\text{true positive}}{\text{true positive + false negative}}$$

using both of these metrics, not sure whether to pick one or the other

instead use F1 Score:
harmonic mean of precision and recall

$$F_1
= \left(\frac{2}{\text{recall}^{-1} + \text{precision}^{-1}}\right)
= 2 \cdot \frac{\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}}$$

or more generally:

$$F_{\beta}
= (1 + \beta^2) \cdot \frac{\text{precision}\cdot\text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$

where $\beta$ is chosen such that recall is considered $\beta$ times as important as precision.

having a well-defined dev set + a single real number evaluation metric allows you to quickly tell which classifier is better and speeds up the iterative process of improving your ML algorithm.

another way is by computing average performance over different metrics (eg different errors in categories)

# satisficing and optimizing metrics

eg:

maximizing accuracy (optimizing metric)

running time <= 100ms (satisficing metric)

# train dev test distributions

datasets can actually impede progress of team

dev aka holdout set

you want your dev and test sets to belong to the same distribution so that we optimize to the "same thing"

choose dev set and test set to reflect data you expect to get in the future and consider important to do well on.

# size of dev and test sets

dev sets: to test out different ideas, to A/B test different algorithms

test sets: to evaluate final classifiers

old paradigm: 70:30 train test
or 60:20:20 train dev test

new paradigm: 1 million training examples: 98:1:1

since deep learning algorithms have such a huge hunger for data, more ought to go towards training

set your test set to be big enough to give high confidence in the overall performance of your system. 10,000 or 100,000 examples might be enough (and might be much less than 30%)

# when to change dev/test sets and metrics

you may realize your target is set in the wrong location midway, in which it is better to move your target than it is to optimize your algorithm

depending on your application, you may want to shift your metrics to focus on false positives or false negatives

$$\frac{1}{m_{dev}}\sum_{i=1}^{m_{dev}}\mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

$$
\frac{1}
{\sum_{i=1}^{m_{dev}}
w^{(i)}}
\sum_{i=1}^{m_{dev}}w^{(i)}\mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

1. define where you want to aim. place the target. how to define a metric to evaluate classifiers
2. worry separately about how to do well on this metric
3. if doing well on your dev/test set does not correspond to doing well on your application, change your metric and/or the dev/test distribution

