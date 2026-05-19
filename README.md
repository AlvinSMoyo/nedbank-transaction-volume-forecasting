# Nedbank Transaction Volume Forecasting Challenge

> Forecasting customer transaction volumes using 18 million banking records, advanced feature engineering, time-aware validation, and blended gradient boosting models.

![Certificate](outputs/zindi_certificate.png)

---

## Executive Summary

This project was developed for the Zindi and Nedbank Transaction Volume Forecasting Challenge, which required participants to predict the total number of transactions each customer would perform during the next three-month period (November 2015 to January 2016).

The solution transformed approximately 18 million transaction records into a customer-level forecasting dataset using extensive behavioural, financial, demographic, and customer lifetime value (CLV) features. Multiple machine learning models were trained and evaluated using fold-specific target encoding, five-fold cross-validation, and ensemble blending.

The final approach combined XGBoost, CatBoost, and LightGBM with optimized blend weights, later extended with neural network diversity, pseudo-labeling, and hill-climbing experiments.

### Official Results

- **Public Leaderboard RMSLE:** 0.390279
- **Private Leaderboard RMSLE:** 0.392521
- **Final Rank:** 166th out of 251 participants
- **Total Submissions:** 61

Several later experiments achieved lower private scores (~0.388 RMSLE), but these were submitted after the final submission selection deadline and did not affect the official ranking.

This project significantly strengthened my practical understanding of feature engineering, forecasting, ensemble learning, pseudo-labeling, and competitive machine learning using real-world banking data.

---

## Competition Objective

The objective was to predict the variable:

- `next_3m_txn_count`

representing the total number of transactions each customer would make during the next three-month period.

### Evaluation Metric

- **RMSLE (Root Mean Squared Logarithmic Error)**

RMSLE penalizes large proportional errors and is particularly well suited to count forecasting problems with highly skewed target distributions.

---

## Dataset Overview

The competition dataset included:

- **18 million transaction records**
- **372,000 financial snapshot rows**
- **11,944 customer profiles**
- **8,360 labeled training customers**
- **3,584 test customers**

Data sources included:

- Transaction history
- Financial balances
- Customer demographics
- Account characteristics

## Feature Engineering Strategy

To transform raw banking records into predictive customer-level features, extensive feature engineering was performed across multiple time windows and behavioral dimensions.

### Core Behavioural Features

Transaction-level data was aggregated over rolling windows of:

- 1 month
- 3 months
- 6 months
- 12 months

For each period, the following metrics were calculated:

- Transaction counts
- Debit counts
- Credit counts
- Total transaction value
- Absolute transaction value
- Mean transaction amount
- Standard deviation (volatility)
- Minimum and maximum balances

### Recency, Frequency, and Monetary (RFM) Signals

The feature set was designed around the classic RFM framework:

- **Recency:** How recently the customer transacted
- **Frequency:** How often the customer transacted
- **Monetary Value:** The value and volatility of transactions

### Momentum and Trend Features

To capture changes in customer behavior over time, several trend indicators were engineered, including:

- Ratios between short- and long-term activity (e.g. `txn_ratio_3m_12m`)
- Transaction acceleration and deceleration
- Rolling growth rates
- Delta features between time windows

### Seasonality Features

Because the prediction window covered the high-variance holiday period (November–January), specific seasonality features were created to measure:

- November to January transaction volatility
- December transaction spikes
- Holiday uplift relative to baseline activity

### Non-Linear Transformations

To better model skewed distributions and diminishing returns, selected features were transformed using:

- Logarithmic transformations (`log1p`)
- Square-root transformations

Examples include:

- `txn_count_3m_log`
- `txn_count_3m_sqrt`
- `debit_count_3m_log`

### Interaction Features

Interaction terms were added to capture relationships between variables, such as:

- `txn_3m_x_age`
- `balance_per_txn`

### Customer Lifetime Value (CLV) Features

To reflect long-term relationship strength, several CLV-style features were introduced:

- `tenure_days`
- `tenure_months`
- `relationship_value`
- `txn_intensity_per_month`
- `debit_intensity_per_month`

### Categorical Encoding

High-cardinality categorical variables such as city, occupation, industry, and certification type were encoded using fold-specific target encoding to reduce leakage and preserve predictive signal.

### Feature Selection

Permutation importance was used to rank features, and the final models were trained using the top ~75 predictors. This reduced noise, improved generalization, and accelerated training.

---

## Modeling Approach

Multiple complementary algorithms were used to capture different aspects of customer behavior.

### Base Models

- **XGBoost** — Robust gradient boosting with strong handling of non-linear interactions
- **CatBoost** — Excellent performance on heterogeneous financial data
- **LightGBM** — Fast and efficient gradient boosting for large feature sets
- **Neural Network** — Included in later experiments to introduce non-tree model diversity

### Validation Strategy

Five-fold cross-validation was used with:

- `log1p` transformation of the target variable
- Fold-specific target encoding
- Out-of-fold (OOF) prediction generation

This ensured alignment with the RMSLE competition metric and minimized target leakage.

### Hyperparameter Optimization

Optuna was used to tune:

- Model-specific hyperparameters
- Ensemble blending weights

### Advanced Experiments

Several advanced techniques were explored:

- Pseudo-labeling using high-confidence test predictions
- Hill-climbing blend optimization
- Neural network diversity
- Controlled prediction clipping

These experiments improved understanding of generalization and ensemble robustness.
