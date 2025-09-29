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
- [Exploratory Data Analysis (EDA)](#exploratory-analysis)
- [Feature Selection](#feature-selection)
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

## Exploratory Data Analysis (EDA)
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
**The proportion of fraudulent transactions: 0.172749%.**

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
<img width="589" height="453" alt="output" src="https://github.com/user-attachments/assets/d19519e1-b858-4c18-87f5-22bf7679448f" />

From this histogram, we can tell:
1. Most Transactions Are Low Value: The biggest bar at the lower left (near log(Amount+1) = 0–1) means most transactions involve very small amounts.
2. Long Tail for Higher Amounts: As the bars decrease to the right, it shows that high-value transactions are rare—consistent with real-world financial data.
3. Distinct Amount Clusters: The peaks at specific log values (e.g., around 1, 2, 3, etc.) suggest common transaction sizes—potentially round numbers or routine purchase values.



Handling class imbalance: Since fraud cases are rare, I use the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to create synthetic fraud samples, ensuring that machine learning models have enough signal to learn from.

Feature selection and scaling: Features are analyzed for relevance, and numerical values are standardized to support algorithms sensitive to scale.

Splitting the data: The dataset is divided into training and test sets to fairly evaluate model performance.


## Feature Selection
Feature selection is vital in machine learning, especially with highly dimensional datasets like the one used here, as it helps reduce noise, minimize overfitting, and improve model interpretability.To boost the performance and reliability of my fraud detection models, I implemented a structured, multi-step feature selection process by implemneting [FeatureSelector](https://github.com/WillKoehrsen/feature-selector) class created by WillKoehrsen to select features to be removed from the  dataset based on the following 5 methods:

1. Remove Features with Excessive Missing Values: Columns with more than 60% missing data were identified and dropped, though our dataset fortunately contained none.
2.  Eliminate Features with Only a Single Unique Value: Such features provide no predictive power and were not present here, but this check helps avoid unnecessary complexity.
3.  Identify and Remove Highly Collinear Features: Features with high correlation (∣r∣>0.97) can introduce redundancy and multicollinearity, but our data did not have such pairs.
4.  Discard Zero-Importance Features: Using a gradient boosting machine (LightGBM), I filtered out features that the model deemed completely irrelevant to the classification task.
5. Prune Low-Importance Features: Finally, I removed features that did not contribute to 99% of cumulative feature importance according to the boosting model.

```python
from feature_selector.feature_selector import FeatureSelector

X = df.drop('Class', axis=1)
y = df['Class']
fs = FeatureSelector(data = X, labels = y)
fs.identify_missing(missing_threshold=0.6)
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.975)
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
low_importance_features[:5]
X_removed = fs.remove(methods = 'all')
```

By systematically applying these techniques, I ensured that my final dataset included only those features most useful for distinguishing fraudulent from legitimate transactions. This thoughtful curation not only increases computational efficiency but also supports the model in learning more meaningful patterns relevant to real-world fraud detection. Feature V2 was removed eventually by implementing these methods.

## Model Training
Since I am dealing with imablanced datasets, which is very common in fraud detection task where the fraudulent cases are extremely rare (0.173% in our dataset), it is important to ensure that both training and testing sets contain a representative proportion of fraudulent cases. That allows the model to learn to recognize fraudulent patterns and evaluate its performance effectively. Therefore, here I apply stratified train-test split alongside a technique called Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class in our dataset.

```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain the ratio of classes in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_removed, y, test_size=0.2, stratify=y, random_state=42)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
counter = Counter(y_train_resampled)
print(counter)
```

I then train the models with both orignal and resampled data, so that I can compare the performance of the models on these data.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(X_train, y_train, X_test, y_test, X_train_resampled, y_train_resampled):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(50, 30, 30, 50), max_iter=500, random_state=42)
    }

    for model_name, model in models.items():
        print(f"\n{model_name} (Original Data):")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'Precision: {precision_score(y_test, y_pred):.4f}')
        print(f'Recall: {recall_score(y_test, y_pred):.4f}')
        print(f'F1 score: {f1_score(y_test, y_pred):.4f}')

        print(f"\n{model_name} (Resampled Data):")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred_resampled = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred_resampled):.4f}')
        print(f'Precision: {precision_score(y_test, y_pred_resampled):.4f}')
        print(f'Recall: {recall_score(y_test, y_pred_resampled):.4f}')
        print(f'F1 score: {f1_score(y_test, y_pred_resampled):.4f}')

evaluate_models(X_train, y_train, X_test, y_test, X_train_resampled, y_train_resampled)
```

## Results

```python
import matplotlib.pyplot as plt

# Storing metrics in a dictionary
results = {
    'Random Forest (Resampled)': {'accuracy': 0.9996, 'precision': 0.8842, 'recall': 0.8571, 'f1_score': 0.8705},
    'Random Forest (Original)': {'accuracy': 0.9996, 'precision': 0.9412, 'recall': 0.8163, 'f1_score': 0.8743},
    'ANN (Original)': {'accuracy': 0.9994, 'precision': 0.8667, 'recall': 0.7959, 'f1_score': 0.8298},
    'ANN (Resampled)': {'accuracy': 0.9989, 'precision': 0.6434, 'recall': 0.8469, 'f1_score': 0.7313},
    'Naive Bayes (Resampled)': {'accuracy': 0.9740, 'precision': 0.0552, 'recall': 0.8776, 'f1_score': 0.1039},
    'Naive Bayes (Original)': {'accuracy': 0.9773, 'precision': 0.0609, 'recall': 0.8469, 'f1_score': 0.1137},
    'Logistic Regression (Resampled)': {'accuracy': 0.9823, 'precision': 0.0827, 'recall': 0.9184, 'f1_score': 0.1518},
    'Logistic Regression (Original)': {'accuracy': 0.9992, 'precision': 0.8228, 'recall': 0.6633, 'f1_score': 0.7345}
}

def plot_metrics(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    num_models = len(results)
    model_names = list(results.keys())
    
    # Set up subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=True)
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        axes[i].barh(model_names, values, color='skyblue')
        axes[i].set_title(metric.capitalize())
        axes[i].set_xlim(0, 1)
        for j, v in enumerate(values):
            axes[i].text(v + 0.01, j, f"{v:.4f}", va='center')  # Add text labels
    
    plt.tight_layout()
    plt.show()

# Call the function to plot the results
plot_metrics(results)
```
<img width="1989" height="490" alt="image" src="https://github.com/user-attachments/assets/c78f17f4-c601-4ef4-862b-2abfae1523dc" />

As a fast growing menace in finance industry, credit card fraud can be detected more effectively by adopting machine learning. In the application supervised learning algorithm in fraud detection, it is important to take the highly skewed datasets into consideration. The presence of significantly more genuine transactions than fraudulent ones (class imbalance) poses a challenge. Specialized techniques like resampling or using class weights are essential to balance the training process.

The reuslts show that 
1. Random Forest with resampled data appears to be the best-performing model for this task, achieving a strong balance between precision, recall, and F1 score. This makes it a practical choice for fraud detection, as it minimizes false positives (high precision) while maintaining a good recall.
3. ANN with original data also performs well and could be a suitable alternative, although it may require further tuning (e.g., different architectures, class weights) to achieve similar robustness as Random Forest.
4. istic Regression could be considered for simpler, smaller datasets but struggles with imbalanced data when resampling is applied.
5. Naive Bayes is not suitable for this type of fraud detection task due to its low precision, which leads to too many false positives.

In summary, Random Forest is likely the best option in this case among these models for detecting fraudulent transactions in an imbalanced dataset, offering strong performance without needing resampling.
