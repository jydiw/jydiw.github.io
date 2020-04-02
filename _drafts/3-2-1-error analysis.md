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

when analyzing mislabeled dev set, add an "incorrectly labeled" column to assess the percent of errors

apply process to dev and test sets to make sure they continue to come from the same distribution

consider examining examples your algorithm got right. maybe it got lucky with an incorrectly labeled example. but it's much easier to validate the mislabels on an accurate algorithm

manually counting the fraction of errors is a very good use of your time in order to prioritize how to optimize your algorithm.

# build quickly, then iterate

how do you pick which to focus on?

if you're tackling a problem for the first time, don't overthink and just build something quick and dirty.

- decide where to set up your target. set up dev/test set and metric
- build initial system quickly
- use bias/variance analysis and error analysis to prioritize your next steps

eg. if you realize that most of your errors in the dev set come from a certain subset of your training set, THEN focus on improving performance in that area. use results of your analysis to prioritize where to go next.

this does not quite apply if you have significant prior experience or if you are drawing from a large body of academic literature in the subject.

