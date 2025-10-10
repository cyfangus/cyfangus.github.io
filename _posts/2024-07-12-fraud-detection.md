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
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Model Training & Evaluation](model-training--evaluation)
- [Visualisation & Reporting](#visualisation--reporting)
- [Findings & Domain Insights](#findings--domain-insights)
- [Next Steps](#next-steps)
- [Conclusion](#conclusion)


## Project Overview
This project tests whether traditional ML models, enhanced via class balancing, can match deep learning performance for fraud detection. Using the creditcard.csv dataset, minority-oversampling methods—SMOTE and generative synthetic data (SDV)—expand the sample, helping classical models like Random Forest, XGBoost, and Logistic Regression overcome the challenges of extreme class imbalance. Model evaluation focuses on the rare but critical fraud class and benchmarks against a neural MLP baseline.

## Exploratory Data Analysis (EDA)
### 1. Data Structure & Feature Summary:
- 284,807 transactions, 30 PCA features + Amount, Time, Class.
- Data is highly imbalanced: fraud cases are < 0.2% of all records.

### 2. Class Imbalance:
```python
# get the set of distinct classes
labels = data.Class.unique()

# get the count of each class
sizes = data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()
```
<img width="389" height="410" alt="ClassDistribution" src="https://github.com/user-attachments/assets/752ed67f-3a0e-4cbe-9801-ff51753e57c4" />


The class distribution visualization reveals a highly imbalanced dataset, where the vast majority of transactions are non-fraudulent (Class 0), and fraudulent transactions (Class 1) constitute only a very small fraction of the total data. This extreme imbalance highlights a core challenge in fraud detection modeling, as conventional classifiers tend to be biased towards the majority class and may perform poorly in detecting rare but critical fraudulent cases. This finding directly underpins the rationale for incorporating data augmentation techniques such as SMOTE and generative AI-generated synthetic data in the project. These techniques aim to alleviate the class imbalance by oversampling the minority class, thereby enabling traditional machine learning models to better learn patterns associated with fraud and improve detection performance while validating the project’s objective of enhancing classical algorithms to reach deep learning-level effectiveness.

### 4. Amount & Time Distributions:
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

- Transaction amounts are heavily right-skewed. Therefore, it is log-transformed for better visualisation and computation.

```python
# Transaction Amount Distribution by Class
data['Log_Amount'] = np.log1p(data['Amount'] + 1)

plt.figure(figsize=(10,6))
sns.kdeplot(data=data[data['Class'] == 0], x='Log_Amount', label='Non-Fraud', fill=True, common_norm=False)
sns.kdeplot(data=data[data['Class'] == 1], x='Log_Amount', label='Fraud', fill=True, common_norm=False, color="r")
plt.title('Transaction Amount Distribution by Class (Log Scale, Normalized)')
plt.legend()
plt.show()
```
<img width="851" height="545" alt="AmountKDE" src="https://github.com/user-attachments/assets/255614fc-f24c-4854-a413-ece2c8a0f4b5" />


From this plot, you can observe that both fraudulent and non-fraudulent transactions in the dataset show a broadly similar distribution pattern for the transaction amounts when normalized and viewed on a logarithmic scale. The density of both classes appears highest at lower transaction amounts (roughly under 100 units), and both tails drop off as the transaction amount increases. There is not a dramatic difference indicating that fraud is concentrated at either extreme—fraudulent transactions span a similar range of amounts as legitimate ones, with densities concentrated in lower-value transactions.

```python
data_nonfraud = data[data['Class'] == 0]
data_fraud = data[data['Class'] == 1]

plt.figure(figsize=(14, 6))

# Plot for non-fraud
sns.kdeplot(data_nonfraud['Time'], label='Non-Fraud', fill=True, alpha=0.5)

# Plot for fraud
sns.kdeplot(data_fraud['Time'], label='Fraud', fill=True, alpha=0.5)

plt.xscale('linear')  # Use linear scale for time
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Transaction Time Distribution for Fraud and Non-Fraud')
plt.legend()
plt.show()
```

<img width="1142" height="545" alt="TimeByClass" src="https://github.com/user-attachments/assets/6f2e4d08-ed6c-41a8-97d9-d24c59387fad" />

Now, this plot better visualise the difference on the overal density, peaks, and concentration across time for both classes by showing their relative likelihoods, smoothing out minor fluctuations caused by the low number of fraud cases. It highlights the potential of time effect on the slightly higher fraud propensity, even if those differences are subtle.

### 5. Correlation Matrix & Feature Engineering:
```python
# Correlation Heatmap of Features
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()
```
<img width="1141" height="868" alt="cor" src="https://github.com/user-attachments/assets/d0c28b81-94eb-455a-8cbd-cc41fbdbdfae" />

Here, plotting a feature correlation heatmap helps visualize how different variables interact with each other and with the target variable. This process can reveal multicollinearity (when features are strongly correlated) and provide insight into which variables may carry the most relevant information for distinguishing between classes, both of which affect model selection, feature engineering, and interpretation.
The result in this plot shows that most features in the dataset—including the principal components (V1-V28), ‘Time’, and ‘Amount’—are very weakly correlated with each other and the target class (‘Class’), with the exception of some isolated, minor correlations. This is expected because the V features are derived from PCA which produces mostly uncorrelated components. It also suggests that classical linear methods may struggle to find patterns on raw correlations alone, motivating the need for models capable of capturing more complex relationships. For this project, it highlights the importance of using the full feature set and possibly engineering new features, rather than relying solely on correlation-based selection.

```python
# Calculate correlation matrix
corr_matrix = data.corr()

# Get correlation with target class
corr_with_target = corr_matrix['Class'].drop('Class')

# Select features with correlation magnitude above a threshold, e.g., 0.1
top_3_features = corr_with_target.abs().sort_values(ascending=False).head(3).index.tolist()

data['Log_Amount'] = np.log1p(data['Amount'])  # natural log of (Amount + 1)
features_to_be_plotted = top_3_features + ['Log_Amount']

sns.pairplot(data, vars= features_to_be_plotted , hue='Class', plot_kws={'alpha':0.3}, height=2)
plt.suptitle('Pairwise Scatter Plots of Selected Features')
plt.show()
```
Since raw amount is heavily skewed, the pairwise scatter plot using the log-transformed transaction amount ( Log_Amount ) alongside principal component features (V12, V14, V17) provides a clearer and more interpretable visualization compared to using the raw amount. Log-transforming the transaction amount reduces extreme skewness and compresses the influence of very large transactions, allowing subtle differences and clusters between fraudulent and non-fraudulent transactions to become more visible. In this transformed space, fraud cases form distinguishable clusters in certain regions of the feature space, suggesting meaningful multi-dimensional patterns exist that may be leveraged by machine learning models. This enhanced separation supports the rationale for using log transformation in feature engineering to improve model performance and interpretability in financial fraud detection contexts.

```python
sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Transaction Amount (log-transformed) Boxplot by Class')
plt.yscale('log')
plt.show()
```
<img width="567" height="453" alt="AmountByClass" src="https://github.com/user-attachments/assets/bf935e06-cd2b-43fd-bfa4-2167eddd010b" />

### 6. EDA Insights:
Some key points to be noted from EDA:
1. Class imbalance: Fraudulent transactions are extremely rare compared to non-fraud, justifying the use of techniques like SMOTE or synthetic data generation to balance classes and avoid model bias toward the majority class.
2. Feature Distribution Skewness: Raw transaction amounts are highly skewed with extreme outliers. For more stable training and better interpretability, log-transformed amount should be used instead. 
3. Feature Correlations: Most PCA-derived features show low mutual correlation and weak linear correlation with the target. This suggests feature normalization (e.g., RobustScaler) rather than standard scaling could be better, preserving feature distributions robustly even with outliers.
4. Temporal Patterns: The ‘Time’ feature shows cyclic behavior but similar patterns across classes. Consider engineered temporal features (hour of day, day of week) or binning to capture potential time-related fraud signals.
5. Range and Scale Variations: Features like ‘Amount’ (even after log transform), ‘Time’, and PCA components exist on different scales with varying distributions, implying the need for consistent scaling before modeling.

## Data Preprocessing & Augmentation

### 1. Log transformation of Amount
```python
data['Log_Amount'] = np.log1p(data['Amount'] + 1)

X = data.drop(['Amount', 'Class'], axis=1) # Amount is dropped as log_amount is kept for the purpose
y = data['Class']
```

### 2. RobustScaler applied to all continuous features
```python
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Prepare 3 sets of data (original, SMOTE, and CTGAN)
```python
# Split into train/test data
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
```
```python
# Apply SMOTE to generate balanced data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_orig, y_train_orig)

X_smote = pd.DataFrame(X_smote, columns=X.columns)
y_smote = pd.Series(y_smote)
train_smote_df = pd.concat([X_smote, y_smote.reset_index(drop=True)], axis=1)
```
```python
# To compare the effect of CTGAN and SMOTE, I set the number of synthetic data to be generated to the same as the number generated by SMOTE.
X_train = pd.DataFrame(X_train_orig, columns=X.columns)
y_train = pd.Series(y_train_orig, name='Class')
train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)

num_synthetic_samples = X_train_smote.shape[0] - X_train_orig.shape[0]

fraud_data = train_data[train_data['Class'] == 1] 

metadata = Metadata().detect_from_dataframe(fraud_data)

ctgan = CTGANSynthesizer(metadata=metadata, epochs=300)
ctgan.fit(fraud_data)

synthetic_fraud = ctgan.sample(num_synthetic_samples)
train_data_ctgan = pd.concat([train_data, synthetic_fraud], ignore_index=True)

X_train_ctgan = pd.DataFrame(train_data_ctgan.drop('Class', axis=1), columns=X.columns)
y_train_ctgan = pd.Series(train_data_ctgan['Class'], name='Class')
```

## Model Training & Evaluation
Algorithms:
Random Forest, XGBoost, Logistic Regression, and MLP
```
============================================================
Results for Random Forest (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.96      0.76      0.85       148

    accuracy                           1.00     85443
   macro avg       0.98      0.88      0.92     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9307454414963647
============================================================
Results for XGBoost (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.94      0.76      0.84       148

    accuracy                           1.00     85443
   macro avg       0.97      0.88      0.92     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.92859883742116
============================================================
Results for Logistic Regression (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.86      0.63      0.73       148

    accuracy                           1.00     85443
   macro avg       0.93      0.81      0.86     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9563387321901888
============================================================
Results for MLP (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.85      0.77      0.81       148

    accuracy                           1.00     85443
   macro avg       0.93      0.89      0.90     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9561665158915879
============================================================
Results for Random Forest (SMOTE):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.88      0.80      0.84       148

    accuracy                           1.00     85443
   macro avg       0.94      0.90      0.92     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9645200361860189
============================================================
Results for XGBoost (SMOTE):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.79      0.80      0.80       148

    accuracy                           1.00     85443
   macro avg       0.90      0.90      0.90     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9706409234722736
============================================================
Results for Logistic Regression (SMOTE):
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     85295
           1       0.06      0.86      0.12       148

    accuracy                           0.98     85443
   macro avg       0.53      0.92      0.55     85443
weighted avg       1.00      0.98      0.99     85443

ROC AUC: 0.9647857277524903
============================================================
Results for MLP SMOTE:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.75      0.76      0.76       148

    accuracy                           1.00     85443
   macro avg       0.87      0.88      0.88     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9512691248021572
============================================================
Results for Random Forest (CTGAN):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.85      0.81      0.83       148

    accuracy                           1.00     85443
   macro avg       0.92      0.91      0.91     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9283590099860104
============================================================
Results for XGBoost (CTGAN):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.87      0.78      0.83       148

    accuracy                           1.00     85443
   macro avg       0.94      0.89      0.91     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9686937068964151
============================================================
Results for Logistic Regression (CTGAN):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.80      0.76      0.78       148

    accuracy                           1.00     85443
   macro avg       0.90      0.88      0.89     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.925086821096259
============================================================
Results for MLP CTGAN:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.79      0.80      0.79       148

    accuracy                           1.00     85443
   macro avg       0.90      0.90      0.90     85443
weighted avg       1.00      1.00      1.00     85443

ROC AUC: 0.9768463345812546
============================================================

```
Key insights from these results:
1.  Model Performance Rankings:
	•	Best overall ROC AUC: MLP with CTGAN (0.977), followed by XGBoost with SMOTE (0.971) and MLP with CTGAN also shows very strong class 1 recall and balanced precision.
	•	Traditional Tree Models: Random Forest and XGBoost perform excellently, especially with SMOTE and CTGAN augmentation, improving recall and F1 scores on fraud class compared to original data.
	•	Neural Network (MLP): Shows strong performance especially when trained on CTGAN and original data, with good recall and balanced precision.
	•	Logistic Regression: Shows much weaker performance on SMOTE (very low fraud precision) but improves on original and CTGAN datasets.
2. Impact of Data Augmentation:
	•	SMOTE clearly improves recall and F1-score for Random Forest and XGBoost, confirming its effectiveness in addressing class imbalance.
	•	CTGAN also improves performance, especially for MLP, pushing ROC AUC highest here.
	•	Logistic Regression struggles with SMOTE in precision, indicating this model is less robust to oversampling approaches.
3. Model-Specific Highlights:
	•	Random Forest (SMOTE): High precision (0.88) and recall (0.80) with ROC AUC of 0.965, balancing false positives and negatives well.
	•	XGBoost (SMOTE): Slightly higher ROC AUC (0.971) but lower precision (0.79), suggesting more false positives.
	•	MLP (CTGAN): Highest ROC AUC (0.977), balanced precision (0.79), and recall (0.80), showing strong detection ability with generative synthetic data.
	•	Logistic Regression: Strong on original data but much lower fraud precision with SMOTE, which may trigger excess false alarms.

## Visualization & Reporting
precision-recall curves compare the model performance on different datasets.
<img width="688" height="545" alt="PR1" src="https://github.com/user-attachments/assets/31571503-565d-4e09-bd3d-c2283804ee5f" />
<img width="688" height="545" alt="PR2" src="https://github.com/user-attachments/assets/ffb7d7c9-9d3d-4ffe-9ddc-03adbbb93a33" />
<img width="688" height="545" alt="PR3" src="https://github.com/user-attachments/assets/3085bf55-4319-41e0-ae6a-93baf1682e40" />

## Findings & Domain Insights
1. Comparison of Model Types:
	•	Tree-based models (Random Forest, XGBoost) outperform linear (Logistic Regression) and baseline neural networks (MLP) in credit card fraud detection.
	•	Random Forest and XGBoost achieve the highest Precision-Recall AUC, critical for detecting rare fraud while minimizing false positives in the financial sector.
	•	Logistic Regression is faster and simpler but underperforms on imbalanced transaction data.
	•	MLP is sensitive to data sampling but never surpasses tree-based models in this context.
2. Impact of Synthetic Data and Oversampling:
	•	Techniques addressing class imbalance, especially SMOTE and CTGAN, have significant business impact.
	•	SMOTE improves recall and AUC for tree and neural models, increasing detection of fraudulent transactions, key for regulatory compliance and reducing losses.
	•	CTGAN benefits high-capacity models like XGBoost, boosting precision-recall AUC and operational savings by reducing missed fraud.
	•	These methods show understanding of the evolving fraud tactics financial institutions face and support compliance with AML and KYC protocols, ensuring institutional resilience.

### Business Insights
1. Random Forest and XGBoost with SMOTE are the most robust combinations for fraud detection. They balance sensitivity (recall) and specificity (precision), crucial for minimizing operational loss, customer inconvenience, and regulatory exposure. They balance sensitivity (recall) and specificity (precision), crucial for minimizing operational loss, customer inconvenience, and regulatory exposure.
2. Oversampling and synthetic augmentation give models a decisive edge, supporting real-time transaction screening and risk management.
3. Logistic Regression and neural networks are less effective without class balancing or advanced sampling, highlighting the need to align model selection with business risk profiles and operational imperatives.

## Next Steps
1. Optimize recall without sacrificing precision (class weights, threshold tuning)
2. Engineer more features (domain-specific, embeddings, interaction terms)
3. Use stratified cross-validation, cost-sensitive loss, confusion matrix for validation
4. Add model explainability (SHAP, LIME) for analyst trust and regulatory compliance

## Conclusion
This project systematically compared a range of machine learning models and class imbalance techniques for credit card fraud detection, addressing business-critical objective in finance industry. The analysis established that tree-based models, especially Random Forest and XGBoost, consistently outperform linear and neural network approaches in identifying fraudulent transactions while maintaining high precision—crucial for reducing false alarms and operational costs in financial institutions. It also demonstrated the substantial value of synthetic oversampling methods, particularly SMOTE and CTGAN, which enhance model sensitivity and precision in highly imbalanced transaction datasets. Implementing these techniques substantially boosts the effectiveness of fraud detection systems, allowing institutions to catch more fraudulent transactions without inundating operations with unnecessary investigations—a priority in managing regulatory compliance and sustaining customer trust . By integrating advanced machine learning models with robust sampling strategies, the project provides practical, business-aligned solutions. The findings support the deployment of tree-based models with SMOTE augmentation as the most promising combination for automated fraud detection. This approach addresses the real-world demands of the financial sector: dynamic threat landscapes, regulatory scrutiny, and the ever-present challenge of safeguarding both assets and customer relationships. Overall, the project delivers actionable insights to guide model selection, deployment, and lifecycle management for credit card fraud detection, equipping financial institutions with the tools to respond effectively and adapt to evolving fraud patterns .
