---
title: "Logistic Regression Basics"
excerpt_separator: "<!--more-->"
categories:
  - DLS
  - data science basics
tags:
  - logistic regression
  - classification
---

# Binary Classification

Just as we see in the linear regression setting, our binary classification (two-class classification) begins with a set of examples $x^{(1)}, x^{(2)}, \dotsc, x^{(m)}$ that we must map to the values of our response variable $y^{(1)}, y^{(2)}, \dotsc, y^{(m)}$. If we were training a model to categorize images as either `cat` or `not cat`, we could attempt to map this model in the following way:

$$y =
\begin{cases}
0 & \quad \text{if not cat}\\
1 & \quad \text{if cat}
\end{cases}
$$

> Ask yourself: What would it mean if our classifier predicted the values $0.5$, $1.2$, and $-0.3$ for our response variable?

Since the response variable is not quantitative, we must figure out how to modify our regression into a classification algorithm.

# Logistic Regression

The logistic regression algorithm connects the worlds of regression and classification.

- **logistic**: uses the logistic function (explained below)
- **regression**: predicts continuous values for the response variable
  - Logistic regression models feed the output of linear regression into the sigmoid function (a special case of the logistic function).
  - It predicts continuous *probabilities* between 0 and 1.
  - We partition these probabilities to then classify our response variable into $0$ or $1$
- Logistic regression is the most common binary classification algorithm.
  - highly interpretable
  - computationally lean
  - mathematically similar to linear regression

## The Sigmoid Function
Rather than modeling the response variable directly, the logistic regression models the *probability* of belonging to the positive case category:

$$\hat{y} = P(\text{cat})$$
or more generally, the probability of our response variable being $1$ given $x$:
$$\hat{y} = P(y = 1\:|\:x)$$

In order to confine our function to range $[0, 1]$, we use the standard logistic function, or **sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

![sigmoid](../assets/images/sigmoid.png)

The function asymptotically approaches $1$ when reaching values of $+\infty$ (and $0$ at negative $-\infty$), allowing us to map all real values of $x$ to the desired range.

## The logistic regression equation
is just the linear regression output fed into the sigmoid function.

$$z = \hat{b} + \sum_{i=1}^{n} \hat{w}_ix_i$$
$$\hat{y} = \sigma(z)$$

## Logistic Regression model optimization is achieved via the Logistic Loss function.

A simple least-squared error can't be used for a logistic regression since our output variable is no longer linear. Predicting $\hat{y} = 0.9$ means $P(y = 1) = 0.9$. This means that $y$ is predicted to be 9 times as likely to be $1$ than $0$. However, $\hat{y} = 0.8$ means $y$ is predicted to be 4 times as likely to be $1$ than $0$; this is because the odds.