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
For the first step of the project, I loaded the 1-million+ transaction records, filtereed out rows with missing key fields (DOB, gender, location, balances), standardised date formats (birthdate, transaction date), and cleaned the rare gender occurence. To have a brief understanding of how the data looks like, I have plotted a few graphs to illustrtae some patterns found in different features.

**1. Gender**
```python
# Distribution of Customer Gender
sns.countplot(data=df, x='CustGender')
plt.title('Customer Gender Distribution')
plt.show()
```

![image](https://github.com/user-attachments/assets/dd09f3a3-30cf-46b9-a1eb-60f9f5ffe9a7)

The data shows that there is an imbalance between gender groups, with males almost 2.5x fold of females.

**2. Locations**
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
![image](https://github.com/user-attachments/assets/2f26fcd6-5e4c-488a-975e-301031ec58dc)

**3. Account Balance and Transaction Amounts**
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
![image](https://github.com/user-attachments/assets/14fd94d5-dbcf-4984-b204-18bef9bbeef5)

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
![image](https://github.com/user-attachments/assets/b12b5508-c0bb-4d8b-b090-a0d1a909fa72)

**4. Age**
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
![image](https://github.com/user-attachments/assets/9e1766ed-58db-48bb-9641-ae93a3cd721f)
Age was computed from the difference between current date and customer's DOB, and then was ploted in a historgram.

**5. EDA Summary:**
- Visualized distributions: account balance and transaction amounts (log-transform to handle skew).
- Grouped top 15 locations (covering 60% of all customers), assigned 'Other' for analysis clarity.
- Filtered out ages <18 and >90 for robust modeling.

## Feature Engineering
After EDA, I then proceeded to feature engineering before applying clustering methods to group customers into 4 segments. These include aggeregate customer activities in terms of their transaction sums, recency, frequency. I also generate interaction features, such as age bins, age-bins-location mix, gender-location mix, age-bins-gender mix, location based GDP per hear as attributes. 
```python
# Group transactions by CustomerID to get aggregate behavioral and value metrics
customer_report = df.groupby('CustomerID').agg({
    'TransactionAmount (INR)': ['sum', 'mean', 'count'],
    'CustAccountBalance': 'last',
    'TransactionDate': ['min', 'max']
})
customer_report.columns = ['TotalTransSum', 'AvgTransAmount', 'TransCount', 'EndBalance', 'FirstTrans', 'LastTrans']

customer_report['Log_TotalTransSum'] = np.log1p(customer_report['TotalTransSum'])
customer_report['Log_AvgTransAmount'] = np.log1p(customer_report['AvgTransAmount'])
customer_report['Log_EndBalance'] = np.log1p(customer_report['EndBalance'])

customer_report['RecencyDays'] = (pd.to_datetime('today') - customer_report['LastTrans']).dt.days
customer_report.reset_index(inplace=True)

features = ['Log_TotalTransSum', 'Log_AvgTransAmount', 'TransCount', 'Log_EndBalance', 'RecencyDays']
scaler = RobustScaler()
X_scaled = scaler.fit_transform(customer_report[features])
```

## Customer Segmentation (K-Means Clustering)
Now, it goes to the main task, using K-means clustering to put customers into 4 segments. The reason why I picked 4 segments and K-means clustering is based on a recent systematic review on algorithmic customer segmentation, that K-means clustering with 4 segments is found to be the most common method in using machine learning methods to group customers. For more details, see [Salminen, J., Mustak, M., Sufyan, M. et al.](https://doi.org/10.1057/s41270-023-00235-5).

```python
kmeans = KMeans(n_clusters=4, random_state=42)
customer_report['Segment'] = kmeans.fit_predict(X_scaled)

segment_profile = customer_report.groupby('Segment').agg({
    'CustomerID': 'count',
    'TotalTransSum': 'sum',
    'AvgTransAmount': 'mean',
    'EndBalance': 'mean'
}).rename(columns={'CustomerID': 'NumOfCustomers'})

# Calculate percentage revenue/profit per segment
total_revenue = segment_profile['TotalTransSum'].sum()
segment_profile['RevenuePct'] = 100 * segment_profile['TotalTransSum'] / total_revenue

print(segment_profile)
```


|        | NumOfCustomers | TotalTransSum | AvgTransAmount   |  EndBalance | RevenuePct  |
|Segment  |----|----|----|----|----|                                                              
0         |       231385 |  1.146041e+09  |   4193.875822 | 228902.678497 |  80.224432  |
1         |       359981 | 1.857673e+08   |   461.943159  | 77088.697022   |13.003959  |
2          |      148792 |  9.890596e+06   |    64.467939  | 46192.802503  | 0.692355 |
3          |       98428  | 8.684481e+07   |   724.731447   |  261.786160  | 6.079254  |  


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
