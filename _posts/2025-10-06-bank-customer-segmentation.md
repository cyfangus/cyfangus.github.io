---
title: 'Bank Customer Segmentation'
date: 2025-10-06
image: https://github.com/user-attachments/assets/848426b4-5e18-4b66-9b06-cc297ead5b66
permalink: /posts/2025/01/bank-customer-segmentation./
tags:  
  - Customer Segmentation
  - KMeans Clustering
  - Light GBM
  - Location- and demographic-based targeting business strategy
---

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/848426b4-5e18-4b66-9b06-cc297ead5b66" />


# Bank Customer Segmentation: Data Science Portfolio Project

## Table of Contents
- [Project Overview](#project-overview)
- [EDA](#eda)
- [Feature Engineering](#feature-engineering)
- [Business & Technical Impact](#business-impact)


## Project Overview

Banks face the challenge of understanding, managing, and retaining millions of customers. This project applies **data-driven segmentation** techniques to classify bank customers according to profitability and behavior, enabling targeted marketing and business strategies that maximize revenue and retention. I use a real-world dataset (Kaggle: 1M+ transactions, 800K+ customers, India) with rich demographics, account balances, and transactional histories. My workflow combines unsupervised clustering (*K-Means*) with demographic analysis and supervised machine learning (LightGBM).

**Project repo:** [GitHub - BankCustomerSegmentation](https://github.com/cyfangus/BankCustomerSegmentation)

---

## EDA
- Loaded 1-million+ transaction records, filtered out rows with missing key fields (DOB, gender, location, balances).
- Standardized date formats (birthdate, transaction date), cleaned rare gender occurrences.

```python
# Distribution of Customer Gender
sns.countplot(data=df, x='CustGender')
plt.title('Customer Gender Distribution')
plt.show()
```

![image](https://github.com/user-attachments/assets/81d5dc29-e9b8-4348-bd7b-717db7d3b52b)
The data shows that there is an imbalance between gender groups, with males almost 2.5x fold of females.

```python
print(f'Customers are from {len(df['CustLocation'].unique())} unqiue locations.')

location_counts = df['CustLocation'].value_counts()
total_customers = len(df)

# Compute cumulative sum of customer counts sorted by top locations
cumulative_pct = location_counts.cumsum() / total_customers * 100

# Print coverage for top 10, 15, 20 locations
for n in [10, 15, 20]:
    print(f"Top {n} locations cover: {cumulative_pct.iloc[n-1]:.2f}% of customers")
```
```
Customers are from 8157 unqiue locations.
Top 10 locations cover: 52.24% of customers
Top 15 locations cover: 59.14% of customers
Top 20 locations cover: 62.93% of customers
```
Because there is a large number of unique locations in the customer records, I decide to keep the top 15 locations and bin the rest into a 'Other' category as it covers about 60% of the customers already.

```python
# Compute and plot top 15 locations by count
top_15_locations = df['CustLocation'].value_counts().nlargest(15).index
df['Top15Location'] = df['CustLocation'].where(df['CustLocation'].isin(top_15_locations), 'Other')

top_15_location_counts = df['Top15Location'].value_counts().sort_values()

plt.figure(figsize=(10,6))
top_15_location_counts.plot(kind='barh', color='skyblue')
plt.title('Top 15 Customer Locations plus Other')
plt.xlabel('Count')
plt.ylabel('CustLocation')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/dcdd3c01-f6a2-4827-9e0b-ae423121d546)

Account Balance and Transaction Amounts were foudn to be heavily skewed, which is very common in real-world scenarios. Log transformation was applied to both to better visualise its distribution.
```python
# Histogram of Customer Account Balance (log-transformed)
df['log_balance'] = np.log1p(df['CustAccountBalance'] + 1)

plt.figure(figsize=(8, 5))
sns.histplot(df['log_balance'], bins=50, kde=True)
plt.title('Log-Transformed Customer Account Balance Distribution')
plt.xlabel('Log(1 + Balance) (INR)')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/a63e321f-79bb-49f3-81fa-99558a33b248)
```python
# Histogram of Transaction Amounts (log-transformed)
df['log_amount'] = np.log1p(df['TransactionAmount (INR)'] +1)

plt.figure(figsize=(8, 5))
sns.histplot(df['log_amount'], bins=50, kde=True)
plt.title('Log-Transformed Transaction Amount Distribution')
plt.xlabel('Log(1 + Amount) (INR)')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/4dcb41f5-aa7d-46c5-8668-102ab712b595)


```python
# Additional: Analyze customer age if needed
if 'CustomerDOB' in df.columns:
    current_date = pd.to_datetime('today').normalize()
    df.loc[df['CustomerDOB'] > current_date, 'CustomerDOB'] -= pd.DateOffset(years=100)
    df['Age'] = (current_date - df['CustomerDOB']).dt.days // 365

sns.histplot(df['Age'], bins=20)
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.show()
```
![Uploading image.png…]()
Age was computed from the difference between current date and customer's DOB, and then was ploted in a historgram.


**EDA Summary:**
- Visualized distributions: account balance and transaction amounts (log-transform to handle skew).
- Grouped top 15 locations (covering 60% of all customers), assigned 'Other' for analysis clarity.
- Filtered out ages <18 and >90 for robust modeling.

## Feature Engineering

- Aggregated activities by customer: total transaction sum, average transaction, frequency, account end balance, recency.
- Generated interaction features: age bins, gender-location mix, GDP-based location attributes.

**4. Customer Segmentation (K-Means Clustering)**

- Applied K-Means (n=4) on standardized behavioral features.
- Identified groups:
    - **Segment 0:** Premier Clients (top 20%, contribute 80%+ revenue)
    - **Segment 1:** Mass Market (largest % of clients, steady mid-level value)
    - **Segment 2:** Dormant/Low Value (least value, disengaged)
    - **Segment 3:** Transactional/Emerging (active, low-average balances)

**5. Demographic Insight**

- Applied Chi-square/Kruskal-Wallis tests to explore segment-demographic relationships.
- Significant patterns found: Segment 0 skewed female and metropolitan, Segment 2/3 skewed male, strong location impact (Mumbai/New Delhi clusters for Premier clients).
- Interaction features using metropolitan GDP/population enrich location profile.

**6. Supervised Prediction (LightGBM)**

- Built a classifier to predict Premier segment status using demographic, behavioral, and engineered features.
- Addressed class imbalance using LightGBM `class_weight`.
- Metrics after class weighting:
    - **Macro F1**: Increased from 0.49 to 0.58
    - **Recall** for top-value customers improved substantially (from 0.09 to 0.58)
    - Balanced precision/recall—model is now much better at catching all valuable customers, key for marketing/retention.

---

## Key Results & Visualization

- **Premier Clients** generate 80%+ of revenue from only 20% of the base—critical for retention focus!
- Segment demographics underpin the business case for personalized offerings.
- *Location-based targeting* revealed: Mumbai/New Delhi dominate Premier segment opportunity.
- *Class weighting* in ML models dramatically improves recall for strategic segment identification.

![Revenue Contribution by Segment](link_to_your_visualization_if_hosted)
![Age Distribution by Segment](link_to_your_visualization_if_hosted)

---

## Business & Technical Impact

- Enables **targeted engagement:** VIP campaigns, mass-market upselling, dormant customer reactivation
- Facilitates resource allocation: banks can prioritize segments and regions that drive the most business impact
- Provides a repeatable framework for clustering and prediction in other domains (retail, insurance, fintech)

---

## Tech Stack

- Python (3.9+), Jupyter Notebook
- Pandas, Numpy, Matplotlib, Seaborn
- scikit-learn, LightGBM, SHAP, SciPy

---

## Source & Next Steps

- Full notebook and code at [GitHub](https://github.com/cyfangus/BankCustomerSegmentation)
- Next steps: extend segmentation to temporal analysis, deploy as real-time dashboard, and apply transfer learning on related banking datasets.

---
