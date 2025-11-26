---
title: 'Beyond the Numbers: How We Used Data Science to Uncover True Crime Hotspots'
date: 2025-11-25
image: https://github.com/user-attachments/assets/f23c2146-7a2b-4039-851d-71cf8a151af0
permalink: /posts/2025/02/crime-mapping/
tags:
  - Crime Mapping
  - Hotspot Policing
  - Crime Hamr Index
  - Time Series Forecasting
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
---

<img width="512" height="512" alt="crime-mapping" src="https://github.com/user-attachments/assets/f23c2146-7a2b-4039-851d-71cf8a151af0" />

# Beyond the Numbers: How We Used Data Science to Uncover True Crime Hotspots

## Table of Contents
- [Introduction](#introduction)
- [Part 1](#part-1)
- [Part 2](#part-2)
- [Part 3](#part-3)
- [Conclusion](#conclusion)


## Introduction
### Why Simple Crime Counts Aren't Enough

For decades, police resources have often been allocated based on simple crime volume. A borough might be flagged as high-risk just because it has a high number of bicycle thefts or shoplifting incidents. But these crimes, while important, don't represent the same level of societal harm as a single robbery or serious assault.

To create truly effective, harm-reduction strategies, we need a smarter approach. Our project shifts the focus from simple volume to calculated Harm, utilizing advanced spatial and time-series modeling to pinpoint where the most damaging crimes cluster and what the future risk looks like.

## Part 1
### Quantifying Harm with the National Crime Harm Index (NCHI)

The first step in this analysis was injecting value into the data. We achieved this by applying the logic of the National Crime Harm Index (NCHI), which quantifies the severity of a crime based on the typical custodial sentence it incurs. For more details, please refer to the concept paper 'Sherman, L.W. and Cambridge University associates., 2020. How to Count Crime: the Cambridge Harm Index Consensus. Cambridge Journal of Evidence-Based Policing, pp.1-14'

Instead of treating every crime equally, we assigned a 'Harm Score' (measured in custodial days) to each incident:

| Crime Type | NCHI Harm Score (Days) | Implication |
| Robbery | 1000 | Highest harm, requiring targeted intervention. |
| Violence/Sexual Offences | 450 | Significant public safety priority. |
| Burglary | 400 | High impact on residents and businesses. |
| Shoplifting | 20 | Low relative harm, high volume. |

By aggregating the data using these weights, we produced a true "Harm Ranking," showing that the top drivers of societal cost were not the most frequent crimes, but the most severe ones (as seen in the accompanying Harm Score Ranking chart). This foundational step ensures our subsequent analysis focuses police time where it saves the most lives and prevents the most serious injury.

<img width="1420" height="1294" alt="CountByCrimeType" src="https://github.com/user-attachments/assets/e9d6a53d-945b-4d6b-9b3c-a755a646ddae" />

<img width="3000" height="2100" alt="harm_score_ranking" src="https://github.com/user-attachments/assets/9e494ffa-aa30-4617-9442-3c16f6af9687" />

## Part 2
### Spatial Clustering—Pinpointing Micro-Hotspots

Traditional hotspot mapping often uses simple kernel density, which can blur the lines between actual micro-hotspots. To provide tactical teams with precise, actionable boundaries, we deployed DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

My Methodology:

Filtering: We filtered the data to include only Ultra-High Harm events (Robbery, Violence, Weapons, etc., with a score >= 300 days).

Clustering: We ran DBSCAN using a small radius (ε = 75 meters) and a minimum threshold (MinPts=10 incidents). This algorithm identifies dense clusters of high-harm events and isolates them from general crime "noise" (outliers).

Visualization: On the interactive map, we visualized the output using Convex Hulls—the smallest possible polygon that enforces all clustered points. This polygon precisely delineates the micro-hotspot boundary.

The Key Insight: Town Centre Nexus

The interactive map clearly shows that these high-harm micro-hotspots are not randomly distributed. They consistently overlap with Wards identified as containing Night-Time Economy (NTE) areas or central commercial zones (highlighted in yellow on the map). This spatial correlation provides clear supporting evidence for targeted police deployments on weekend nights and specific commercial security checks.

While the map might be too large to be shared, I have taken a screenshot here to show you how it works.
<img width="2543" height="1287" alt="Screenshot 2025-11-26 at 16 13 10" src="https://github.com/user-attachments/assets/4b918e41-3be1-4e62-ae5f-73396b26b031" />


## Part 3
### Predictive Analytics—Forecasting Future Harm

To move from reactive to proactive policing, we must predict future risk. We used Time Series Forecasting to model the overall trend in total crime harm.

Using the Exponential Triple Smoothing (ETS) model—a robust algorithm that accounts for both trend and seasonality—we forecasted the total NCHI harm score for the next 12 months (as shown in the Harm Forecast chart).

Impact: This forecast allows police command to:

Budgeting: Allocate resources for high-harm initiatives well in advance.

Proactive Planning: If the forecast shows an expected increase in total harm, preventative operations can be planned immediately, rather than waiting for quarterly incident reviews.

<img width="3000" height="1800" alt="harm_trends" src="https://github.com/user-attachments/assets/566b887c-171f-40e6-b68f-a8c81f055d15" />
<img width="3000" height="1800" alt="harm_forecast" src="https://github.com/user-attachments/assets/78e4c63f-cf09-46e9-94dd-e29a131e99e0" />



## Conclusion
### A Data-Driven Approach to Public Safety

This project demonstrates how combining modern data science techniques (NCHI scoring, DBSCAN clustering, and Time Series Forecasting) with open public data can transform policing. We moved beyond simple volume metrics to:

Focus on Harm: Prioritizing crimes that cost society the most.

Target Hotspots: Defining precise boundaries for micro-hotspots linked to high-impact locations like town centres.

Predict Risk: Giving leaders a 12-month outlook on future harm levels.

This model provides an efficient, evidence-based roadmap for reducing the most damaging types of crime in our community.
