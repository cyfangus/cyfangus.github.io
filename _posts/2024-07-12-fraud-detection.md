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
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()
```
<img width="591" height="453" alt="ClassDistribution" src="https://github.com/user-attachments/assets/6cb23748-38ea-4789-9291-1e2e91175791" />
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

### 3. SMOTE for interpolating minority samples
```python
# Split into train/test data
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to generate balanced data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_orig, y_train_orig)

X_smote = pd.DataFrame(X_smote, columns=X.columns)
y_smote = pd.Series(y_smote)
train_smote_df = pd.concat([X_smote, y_smote.reset_index(drop=True)], axis=1)
```

### 4. Synthetic Data Vault (SDV) for creating new data from probabilistic modeling, blending with SMOTE records for richer distribution
```python
# Take smaller random sample (e.g., 10-20%) of SMOTE data for SDV training
sample_frac = 0.1
train_sdv_sample = train_smote_df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

# Fit GaussianCopula on SMOTE training data including target column
metadata = Metadata.detect_from_dataframe(
    data=train_smote_df,
    table_name='fraud_data')

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=train_sdv_sample)

# Generate synthetic data samples as needed, e.g., 20% of SMOTE size
num_samples = int(0.2 * len(train_smote_df))
synthetic_data = synthesizer.sample(num_samples)

# Now combine SMOTE data with SDV synthetic data for final training set if desired
X_train_syn = pd.concat([X_smote, synthetic_data.drop('Class', axis=1)], ignore_index=True)
y_train_syn = pd.concat([y_smote.reset_index(drop=True), synthetic_data['Class'].reset_index(drop=True)], ignore_index=True)
```

## Model Training & Evaluation
Algorithms:
Random Forest, XGBoost, Logistic Regression, and MLP
```
============================================================
Results for Random Forest (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.97      0.77      0.86        98

    accuracy                           1.00     56962
   macro avg       0.99      0.88      0.93     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.9476154347501522
============================================================
Results for XGBoost (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.94      0.81      0.87        98

    accuracy                           1.00     56962
   macro avg       0.97      0.90      0.93     5696
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.940590259035522
============================================================
Results for Logistic Regression (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.85      0.58      0.69        98

    accuracy                           1.00     56962
   macro avg       0.93      0.79      0.85     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.9736338331055552
============================================================
Results for MLP (Original):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.86      0.80      0.83        98

    accuracy                           1.00     56962
   macro avg       0.93      0.90      0.91     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.9638699352841869
============================================================
Results for Random Forest (Synthetic):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.87      0.85      0.86        98

    accuracy                           1.00     56962
   macro avg       0.94      0.92      0.93     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.9844630188175439
============================================================
Results for XGBoost (Synthetic):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.48      0.89      0.62        98

    accuracy                           1.00     56962
   macro avg       0.74      0.94      0.81     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.992670661399056
============================================================
Results for Logistic Regression (Synthetic):
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     56864
           1       0.09      0.91      0.17        98

    accuracy                           0.98     56962
   macro avg       0.55      0.95      0.58     56962
weighted avg       1.00      0.98      0.99     56962

ROC AUC: 0.9776416053196744
============================================================

Results for MLP (Synthetic):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.57      0.88      0.69        98

    accuracy                           1.00     56962
   macro avg       0.78      0.94      0.85     56962
weighted avg       1.00      1.00      1.00     56962

ROC AUC: 0.9664051643448599
============================================================

```
Key insights from these results:
1. Synthetic data and SMOTE dramatically improve recall across models, especially for minority (fraud) cases:
For Random Forest, recall jumps from 0.77 to 0.85, XGBoost from 0.81 to 0.89, and Logistic Regression from 0.58 to 0.91 after augmentation—demonstrating that the models are much better at detecting fraud when the class is rebalanced. ROC AUC scores also increase, signaling greater separability between classes.
2. There is a precision-recall trade-off, especially in XGBoost and Logistic Regression after synthetic augmentation:
While recall for the fraud class improves, precision drops—XGBoost precision falls from 0.94 to 0.48, and Logistic Regression from 0.85 to 0.09, meaning more false positives (non-fraud flagged as fraud). This is critical for operational settings: High recall means fewer missed frauds, but more customers might be inconvenienced by false alarms.
3. Traditional ML (Random Forest, XGBoost) and MLP outperform Logistic Regression (Linear model) for fraud detection post-augmentation:
Logistic Regression’s recall spikes with synthetic samples but its precision collapses, casting doubt on its practical usability. In contrast, Random Forest maintains balanced precision and recall (0.87/0.85), while XGBoost and MLP provide high recall potential for aggressive fraud detection strategies. Model selection should balance business cost and tolerance for investigation workload.

## Visualization & Reporting
ROC curves compare original vs synthetic-augmented data for each model.
<img width="1008" height="699" alt="ROC_Orig" src="https://github.com/user-attachments/assets/10dc238a-7078-4324-bee6-6c0a6e968cfc" />
<img width="1008" height="699" alt="ROC_syn" src="https://github.com/user-attachments/assets/39f9360c-91c4-4a29-8eff-b5235597f522" />
<img width="1008" height="699" alt="ROC_MLP" src="https://github.com/user-attachments/assets/aec252f6-4b6e-4398-9078-15883d6552e3" />
1. Synthetic data boosts overall model performance, especially for minority class (fraud):
The ROC curves show higher AUC values on synthetic data (e.g., XGBoost Synthetic AUC=0.993, Random Forest Synthetic AUC=0.984) compared to the original data (XGBoost Original AUC=0.941, Random Forest Original AUC=0.948). This means the models are better able to distinguish fraud from non-fraud when supplied with balanced, synthetic samples.
2. MLP, XGBoost, and Random Forest are all highly effective classifiers but synthetic augmentation provides the most dramatic benefit for XGBoost:
On the synthetic curve plot, XGBoost’s curve nearly hugs the top left corner, indicating extremely strong performance (few false positives at nearly all TPR thresholds). MLP and Random Forest are also strong, but the increment is sharpest for XGBoost.
3. All models outperform the random baseline and logistic regression remains competitive only in AUC metric—not in precision-recall:
All ROC curves lie well above the red dashed line (random) on both data sources. Logistic Regression maintains a high AUC but, as earlier results showed, its practical fraud detection usability is limited due to drops in precision on synthetic samples.

## Findings & Domain Insights
1. Synthetic sampling boosts recall, critical for fraud loss prevention.
2. Models must be tuned for cost—lots of false positives can overload investigators.
3. XGBoost favored for recall, Random Forest/MLP for balanced deployment.
4. Model choice should reflect business tolerances, not just metrics.

## Next Steps
1. Optimize recall without sacrificing precision (class weights, threshold tuning)
2. Engineer more features (domain-specific, embeddings, interaction terms)
3. Use stratified cross-validation, cost-sensitive loss, confusion matrix for validation
4. Add model explainability (SHAP, LIME) for analyst trust and regulatory compliance

## Conclusion
Class balancing is essential for fraud detection—synthetic data unlocks the ability for classic models to catch elusive fraud. The right ML strategy can prevent millions in losses, but requires careful metrics, business context understanding, and operational tuning.
