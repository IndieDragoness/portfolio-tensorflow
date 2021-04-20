# Portfolio Tensorflow Documentation
This is a combination of notes and code from my Tensorflow crash course, and is a part of my portfolio project to demonstrate
capability.

# Table of Contents
* [How to Approach a Tensorflow Machine Learning Project](#how-to-approach-a-tensorflow-machine-learning-project)
  * [Problem Framing](#problem-framing)
  * [Data Prep](#data-prep)
  * [Testing and Debugging in Machine Learning](#testing-and-debugging-in-machine-learning)
* [Types of Machine Learning](#types-of-machine-learning)
  * [Clustering Systems](#clustering-systems)
  * [Recommendation Systems](#recommendation-systems)
  * [GANs](#gans)
  * [Deep Q](#deep-q)

* [Real World Examples](#real-world-examples)

* [Coursework](#coursework)
  * [Categories of Machine Learning](#categories-of-machine-learning)
  * [Linear Regression](#linear-regression)
  * [Training and Loss](#training-and-loss)
	* [Squared Loss Function](#squared-loss-function)
  * [Reducing Loss Iteratively](#reducing-loss-iteratively)

* [Glossary](#glossary)

# How to Approach a Tensorflow Machine Learning Project
Google has great step-by-step walkthroughs for solving common ML problems using best practices: [Machine Learning Guides](https://developers.google.com/machine-learning/guides)

## Problem Framing
Sources:
* [Crash Course Framing](https://developers.google.com/machine-learning/crash-course/framing/)
* [Full Introduction to Framing](https://developers.google.com/machine-learning/problem-framing)

## Data Prep
Sources:
* [Data Preparation and Feature Engineering in ML](https://developers.google.com/machine-learning/data-prep)

## Testing and Debugging in Machine Learning
Sources:
* [Testing and Debugging in Machine Learning](https://developers.google.com/machine-learning/testing-debugging)

# Types of Machine Learning

## Clustering Systems
Sources:
* [Clustering](https://developers.google.com/machine-learning/clustering)

## Recommendation Systems
Sources:
* [Recommendation Systems](https://developers.google.com/machine-learning/recommendation)

## GANs
Sources:
* [Generative Adversarial Networks (GANs)](https://developers.google.com/machine-learning/gan)

## Deep Q
Sources:
* [Deep Q Learning](https://www.mlq.ai/deep-reinforcement-learning-q-learning/)

Q-learning works well when we have a relatively simple environment to solve, but when the number of states and actions we can take gets more complex we use deep learning as a function approximator.

# Real World Examples
Google has good examples: [Machine Learning Practica](https://developers.google.com/machine-learning/practica)

# Coursework

## Linear Regression
Similar to `y = m x + b` except in machine learning, we use `y=wx+b` where `w` is the ML Weights. This function 
creates a line on 2D Axes that attempts to best match the relationship between the data.

* `y` is the value we're trying to predict.
* `m` is the slope of the line.
* `x` is the value of our input feature.
* `b` is the y-intercept.

In Machine Learning we write it a little differently: `y' = b + w1 x1`

* `y'` is the predicted label (a desired output).
* `b` is the bias (y-intercept), sometimes referred to as `w0`.
* `w1` is the weight of feature 1. Weight is the same concept as the 'slope' `m` in the traditional equation of the line.
* `x1` is a feature (a known input).

To **infer** (predict) the temperature `y'` for a new value `x1`, just substitute `x1` value into this model.

Even though this only used one feature, a more sophisticated model might have multiple features with separate weights:

`y' = b + w1 x1 + w2 x2 + w3 x3`

## Training and Loss
Training simply means determining good values for all the weights and the bias from labeled examples. In Supervised Learning,
a machine learning algorithm builds a model by examining examples and minimizing loss; this is called Emperical Risk Minimization.

Loss is the penalty for a bad prediction. There's single example loss, and dataset-wide loss. Zero is perfect (no loss). The goal
of training is to find a set of weights and biases with low loss, on average, across all examples.

### Squared Loss Function
Squared loss is a popular loss function (also known as L2 loss). Squared loss for a single example is:
```
= the square of the difference between the label and the prediction
= (observation - prediction(x)) ^ 2
= (y -y') ^ 2
```

**Mean Square Error (MSE)**: Is the average squared loss per example over the whole dataset. To calculate, sum up all squared
losses for individual examples and then divide by the number of examples. [Click here to see the equation](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss).

NOTE: Data points get exponentially higher loss the further they deviate. This means even a couple datapoints could have much 
more loss than many, slightly off data points.

## Reducing Loss Iteratively
To train a model, you need a good way to reduce model loss. An iterative approach is one widely used method for reducing loss;
think 'walking down a hill' or the 'Hot and Cold' game. Iterative strategies are prevalent in ML because they scale well to large
data sets.

![Gradient Descent](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

# Glossary
Google has a good Glossary: [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
