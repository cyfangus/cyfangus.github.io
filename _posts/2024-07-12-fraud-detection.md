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
from collections import Counter

df = pd.read_csv('creditcard.csv')
counter = Counter(df['Class'])
print(f'The proportion of fraudulent transactions: {100*counter[1]/len(df):3f}%')
```
**The proportion of fraudulent transactions: 0.172749%**
With this code, I confirmed that the dataset has a class imbalance with only 0.17% labelled as frauduelent transactions.

Next, I move on to explore the distribution of Time and Amount of the transanction records.
```python
# Time is originally stored as the unit of seconds, therefore divide 60*60 to transform into hour
df['Hour'] = df['Time']/(60*60)
plt.figure(figsize=(10,6))
plt.hist(df['Hour'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Hours since first transaction')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Transactions Over Time')
plt.show()
```
![image](https://github.com/user-attachments/assets/9b9e1d23-dca2-48d2-8abe-b9a83b44700e)

With this historgram, I now can tell how transactions are distributed across the two-day period. The histogram shows a higher frequency of transactions during typical waking/working hours, with fewer transactions overnight. Since the dataset covers two consecutive days, you may see two similar patterns or peaks, relating to activity across both days. 
From a fraud analysis perspective, this concentration can:
1. Help detect anomalous transactions that occur at odd hours (e.g., late at night or early morning), which may be more likely to be fraudulent.
2. Inform model feature engineering, e.g., by binning or encoding "time of day" to capture riskier time windows.
 
 ```python
df['Amount'].describe()
```
count,284807.000000
mean,88.349619
std,250.120109
min,0.000000
25%,5.600000
50%,22.000000
75%,77.165000
max,25691.160000

From the above descriptive statistics, we can see the data is hevaily skewed. Therefore, log transformation will allow us to better visualize these patterns and prepare the feature for modeling (reducing skew, stabilizing variance).

```python
import numpy as np

plt.hist(np.log1p(df['Amount']), bins=50)
plt.title('Histogram of Log-Transformed Transaction Amounts')
plt.xlabel('Log(Amount + 1)')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/99ed6bd3-9875-4e51-80b8-f18102ba8161)
From this histogram, we can tell:
1. Most Transactions Are Low Value: The biggest bar at the lower left (near log(Amount+1) = 0–1) means most transactions involve very small amounts.
2. Long Tail for Higher Amounts: As the bars decrease to the right, it shows that high-value transactions are rare—consistent with real-world financial data.
3. Distinct Amount Clusters: The peaks at specific log values (e.g., around 1, 2, 3, etc.) suggest common transaction sizes—potentially round numbers or routine purchase values.
4. 

Handling class imbalance: Since fraud cases are rare, I use the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to create synthetic fraud samples, ensuring that machine learning models have enough signal to learn from.

Feature selection and scaling: Features are analyzed for relevance, and numerical values are standardized to support algorithms sensitive to scale.

Splitting the data: The dataset is divided into training and test sets to fairly evaluate model performance.


## Exploratory Analysis

## Model Training

## Results
