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

# error analysis

cat classifier:

- high false positives
- should you train your cat classifier to do better on dogs?

instead, take a sample of your mislabeled dev set examples
- analyze the misclassifications
- assess the "ceiling" of performance
- figure out where to go from there
- if 50% were dog images, then it would be worth exploring this route

evaluate multiple ideas in parallel
- fix pictures of dogs
- fix great cats
- improve performance on blurry images

spreadsheet with images as each row, ideas (dog, great cats, blurry) for each column, mark 1 if the image pertains to that column. find percentage to triage your improvements and deploy different teams.

# cleaning up incorrectly labeled data

mislabeled example: algorithm incorrectly predicts the example

incorrectly labeled example: the data is incorrectly labeled

deep learning algorithms are quite robust to **random** errors in the training set, but not systematic errors

when analyzing mislabeled dev set, add an "incorrectly labeled" column