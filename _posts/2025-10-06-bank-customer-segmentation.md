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
- [Workflow Overview](#workflow-overview)
- [Key Results & Visualization](#key-result)
- [Business & Technical Impact](#business-impact)


## Project Overview

Banks face the challenge of understanding, managing, and retaining millions of customers. This project applies **data-driven segmentation** techniques to classify bank customers according to profitability and behavior, enabling targeted marketing and business strategies that maximize revenue and retention. 
**Project repo:** [GitHub - BankCustomerSegmentation](https://github.com/cyfangus/BankCustomerSegmentation)
I use a real-world dataset (Kaggle: 1M+ transactions, 800K+ customers, India) with rich demographics, account balances, and transactional histories. My workflow combines unsupervised clustering (*K-Means*) with demographic analysis and supervised machine learning (LightGBM).

---

## Workflow Overview

**1. Data Loading & Cleansing**

- Loaded 1-million+ transaction records, filtered out rows with missing key fields (DOB, gender, location, balances).
- Standardized date formats (birthdate, transaction date), cleaned rare gender occurrences.

**2. Exploratory Data Analysis**

- Visualized distributions: account balance and transaction amounts (log-transform to handle skew).
- Grouped top 15 locations (covering 60% of all customers), assigned 'Other' for analysis clarity.
- Filtered out ages <18 and >90 for robust modeling.

**3. Feature Engineering**

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
