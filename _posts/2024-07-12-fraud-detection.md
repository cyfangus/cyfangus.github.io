---
title: 'Credit Card Fraud Detection: Traditional ML Enhanced with Synthetic Data & SMOTE'
date: 2024-07-12
image: https://github.com/user-attachments/assets/8262188c-208d-4f80-8c46-572ad47b8711
permalink: /posts/2024/04/fraud-detection/
tags:
  - Synthetic Minority Over-sampling Technique (SMOTE)
  - Random Forest
  - Logsitic Regression
  - XGBoost
  - Neural Network
---

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/8262188c-208d-4f80-8c46-572ad47b8711" />

# Credit Card Fraud Detection: Traditional ML Enhanced with Synthetic Data & SMOTE

## Table of Contents
- [Project Overview](#project-overview)
- [EDA](#eda)
- [Data Preprocessing & Augmentation](#data-preprocessing-&-augmentation)
- [Model Training & Evaluation](model-training-&-evaluation)
- [Visualisation & Reporting](#visualisation-&-reporting)
- [Findings & Domain Insights](#findings-&-domain-insights)
- [Next Steps](#next-steps)


## Project Overview
This project tests whether traditional ML models, enhanced via class balancing, can match deep learning performance for fraud detection. Using the creditcard.csv dataset, minority-oversampling methods—SMOTE and generative synthetic data (SDV)—expand the sample, helping classical models like Random Forest, XGBoost, and Logistic Regression overcome the challenges of extreme class imbalance. Model evaluation focuses on the rare but critical fraud class and benchmarks against a neural MLP baseline.

## Exploratory Data Analysis (EDA)
1. Data Structure & Feature Summary:
- 284,807 transactions, 30 PCA features + Amount, Time, Class.
- Data is highly imbalanced: fraud cases are < 0.2% of all records.

2. Class Imbalance:
```python
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()
```
<img width="591" height="453" alt="ClassDistribution" src="https://github.com/user-attachments/assets/6cb23748-38ea-4789-9291-1e2e91175791" />
The class distribution visualization reveals a highly imbalanced dataset, where the vast majority of transactions are non-fraudulent (Class 0), and fraudulent transactions (Class 1) constitute only a very small fraction of the total data. This extreme imbalance highlights a core challenge in fraud detection modeling, as conventional classifiers tend to be biased towards the majority class and may perform poorly in detecting rare but critical fraudulent cases. This finding directly underpins the rationale for incorporating data augmentation techniques such as SMOTE and generative AI-generated synthetic data in the project. These techniques aim to alleviate the class imbalance by oversampling the minority class, thereby enabling traditional machine learning models to better learn patterns associated with fraud and improve detection performance while validating the project’s objective of enhancing classical algorithms to reach deep learning-level effectiveness.

4. Amount & Time Distributions:
- Transaction amounts are heavily right-skewed. Therefore, a KDE plot is used to compare the density of fraud and non-fraud transaction.
```python
# Transaction Amount Distribution by Class
plt.figure(figsize=(10,6))
sns.kdeplot(data=data[data['Class'] == 0], x='Amount', label='Non-Fraud', fill=True, common_norm=False)
sns.kdeplot(data=data[data['Class'] == 1], x='Amount', label='Fraud', fill=True, common_norm=False, color="r")
plt.xscale('log')
plt.title('Transaction Amount Distribution by Class (Log Scale, Normalized)')
plt.legend()
plt.show()
```
<img width="866" height="553" alt="TransactionByClass" src="https://github.com/user-attachments/assets/c999eb30-013f-4a0f-b881-f8bc564b407b" />
From this plot, you can observe that both fraudulent and non-fraudulent transactions in the dataset show a broadly similar distribution pattern for the transaction amounts when normalized and viewed on a logarithmic scale. The density of both classes appears highest at lower transaction amounts (roughly under 100 units), and both tails drop off as the transaction amount increases. There is not a dramatic difference indicating that fraud is concentrated at either extreme—fraudulent transactions span a similar range of amounts as legitimate ones, with densities concentrated in lower-value transactions.
- Time-based features show periodic trends but fraud mirrors the overall transaction cycles.

Correlation Matrix & Feature Engineering:

Features are mainly uncorrelated (PCA), so non-linear or ensemble models are needed.

Log-transforming Amount improves interpretability; engineered temporal features suggested.

EDA Insights:

Outlines risks of ignoring minority class, need for robust scaling, and value of combining multiple features.

3. Data Preprocessing & Augmentation
Log transformation of Amount

RobustScaler applied to all continuous features

SMOTE for interpolating minority samples

Synthetic Data Vault (SDV) for creating new data from probabilistic modeling, blending with SMOTE records for richer distribution

Balanced training sets maximize model learning on fraud records

4. Model Training & Evaluation
Algorithms:

Random Forest, XGBoost, Logistic Regression, and MLP

Hyperparameters tuned via cross-validation

Metrics:

Precision, recall, F1, ROC-AUC—especially focused on their values for Class 1 (fraud).

Comparative Results

On original data: Good AUC, moderate recall for minority class.

On synthetic data: AUC and recall for fraud improve; sometimes precision drops (more false positives), especially for Logistic Regression—acceptable when false negatives are costlier than reviews.

5. Visualization & Reporting
ROC curves compare original vs synthetic-augmented data for each model.

Precision/recall tables help interpret results operationally.

Visuals support arguments for model selection—MLP and Random Forest give a precision/recall balance, XGBoost shines at recall.

6. Findings & Domain Insights
Synthetic sampling boosts recall, critical for fraud loss prevention.

Models must be tuned for cost—lots of false positives can overload investigators.

XGBoost favored for recall, Random Forest/MLP for balanced deployment.

Model choice should reflect business tolerances, not just metrics.

7. Next Steps
Optimize recall without sacrificing precision (class weights, threshold tuning)

Engineer more features (domain-specific, embeddings, interaction terms)

Use stratified cross-validation, cost-sensitive loss, confusion matrix for validation

Add model explainability (SHAP, LIME) for analyst trust and regulatory compliance

8. Conclusion
Class balancing is essential for fraud detection—synthetic data unlocks the ability for classic models to catch elusive fraud. The right ML strategy can prevent millions in losses, but requires careful metrics, business context understanding, and operational tuning.





=====================
# Fraud Detection Using Supervised Learning with Synthetic Minority Over-sampling Technique (SMOTE) & Generative Synthetic Data
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


For more details, please refer to the corresponding repository [Fraud Detection with Supervised Learning (Logistic Regression, XGBoost, Neural Network)](https://github.com/cyfangus/fraud_detection_supervised_learning).

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
<img width="871" height="545" alt="output1" src="https://github.com/user-attachments/assets/d0da776b-70f6-4b30-aa1b-14f6bc3e6eef" />

With this historgram, I now can tell how transactions are distributed across the two-day period. The histogram shows a higher frequency of transactions during typical waking/working hours, with fewer transactions overnight. Since the dataset covers two consecutive days, you may see two similar patterns or peaks, relating to activity across both days. 
From a fraud analysis perspective, this concentration can:
1. Help detect anomalous transactions that occur at odd hours (e.g., late at night or early morning), which may be more likely to be fraudulent.
2. Inform model feature engineering, e.g., by binning or encoding "time of day" to capture riskier time windows.
 
 ```python
df['Amount'].describe()
```

| count | 284807.000000 |
| mean | 88.349619 |
| std | 250.120109 |
| min | 0.000000 |
| 25% | 5.600000 |
| 50% | 22.000000 |
| 75% | 77.165000 |
| max | 25691.160000 |


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

## Feature Selection
Feature selection is vital in machine learning, especially with highly dimensional datasets like the one used here, as it helps reduce noise, minimize overfitting, and improve model interpretability. To boost the performance and reliability of my fraud detection models, I implemented a structured, multi-step feature selection process by implemneting [FeatureSelector](https://github.com/WillKoehrsen/feature-selector) class created by WillKoehrsen to select features to be removed from the  dataset based on the following 5 methods:

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
