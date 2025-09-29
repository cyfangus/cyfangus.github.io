---
title: 'Fraud detection with supervised learning'
date: 2024-07-12
permalink: /posts/2024/04/fraud-detection/
tags:
  - Synthetic Minority Over-sampling Technique (SMOTE)
  - Logsitic Regression
  - XGBoost
  - Neural Network
---
## Table of Contents

- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Training](#model-training)
- [Results](#results)


## Project Overview

This project explores credit card fraud detection using a range of supervised learning techniques. Leveraging a publicly available, highly imbalanced dataset of European credit card transactions, the goal is to identify fraudulent activity—an especially challenging problem in finance.

To tackle this, I compare algorithms including Logistic Regression, Random Forest, Naïve Bayes, and Neural Networks. I address class imbalance with the SMOTE oversampling method, and benchmark results against published research to ensure reliability.

The project demonstrates:
1. How machine learning can uncover fraud patterns in real-world financial data
2. Practical challenges like data imbalance and model interpretability
    Actionable model comparisons tied to business needs

If you’re interested in the technical details, open-source notebook, or further reading, see the links provided below. This post aims to share hands-on insights and lessons learned from building a fraud detection pipeline—making it accessible for anyone exploring data science for crime prevention or financial security.


For more details, please refer to the corresponding repository [Fraud Detection with Supervised Learning (Logistic Regression, XGBoost, Neural Network)](https://github.com/cyfangus/fraud_detection_supervised_learning)).

## Data Preparation
The analysis begins with a [real-world Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) containing over 280,000 credit card transactions, of which only 492 are fraudulent. This dataset is both anonymized for privacy and highly imbalanced, making it a realistic but challenging test case for fraud detection.

Key data preparation steps include:

Loading and inspecting the dataset: Checking for missing values, exploring class distribution, and getting familiar with the anonymized feature set.
```python
import pandas as pd

df = pd.read_csv('creditcard.csv')
print(df.info())
print(df['Class'].value_counts())
```

With this code, I confirmed that the dataset has 284,807 rows and observed the class imbalance.

Handling class imbalance: Since fraud cases are rare, I use the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to create synthetic fraud samples, ensuring that machine learning models have enough signal to learn from.

Feature selection and scaling: Features are analyzed for relevance, and numerical values are standardized to support algorithms sensitive to scale.

Splitting the data: The dataset is divided into training and test sets to fairly evaluate model performance.


## Exploratory Analysis

## Model Training

## Results
