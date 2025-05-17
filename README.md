
# 🔍 NYC Neighborhood Safety Classification Using Machine Learning (2010–2024)

This project explores whether machine learning can accurately classify neighborhood safety in New York City using arrest and shooting data from 2010 to 2024. After applying **PCA** and **K-Means clustering** to define safety labels on the arrest data, these safety labels were then transferred into the shooting data at the same granularity. Two supervised models — **Random Forest** and **Artificial Neural Network (ANN)** — were trained to predict one of two labels:
- **Unsafe and Vulnerable to Shooting**
- **Safe but Vulnerable to Shooting**

Results showed **high predictive accuracy** and revealed key predictors such as jurisdiction code, time of day, holiday indicators, and proximity to major infrastructure. These findings offer **actionable insights for data-informed policy** aimed at improving public safety in NYC.

## Data Sources
![image](https://github.com/user-attachments/assets/1197759b-0f87-4f02-95a0-7a7f009741b6)
- NYPD Arrest Data (2010-2024): https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data
- NYPD Shooting Data (2010-2024): https://data.cityofnewyork.us/Public-Safety/NYPD-Shooting-Incident-Data-Historic-/833y-fsy8/about_data

## Project Notebooks

### 🔹 [Crime Data Clustering](Crime%20Data%20Clustering.ipynb)

- Uses PCA and K-Means to cluster NYC precincts based on arrest data
- Derives initial safety labels ("Safe but Vulnerable", "Unsafe") from arrest features
- Outputs clustered dataset for integration with shooting records

### 🔹 [Shooting Data Processing](Shooting%20Data%20Processing.ipynb)

- Loads and cleans NYPD shooting incident data (2010–2024)
- Performs extensive feature engineering: Holiday flags, temperature bands, proximity indicators
- Merges clustering-based labels for supervised learning

### 🔹 [Shooting Data Prediction](Shooting%20Data%20Prediction.ipynb)

- Trains and evaluates two classifiers: Random Forest and ANN
- Conducts hyperparameter tuning via GridSearchCV
- Outputs: Confusion matrices, Feature importances

> <sub>⚠️ Note: Section links within notebooks may not work directly on GitHub. For full navigation, open notebooks in Jupyter or [nbviewer.org](https://nbviewer.org).</sub>

---

## Tools & Libraries
- Python (Scikit-learn, Pandas, NumPy, Matplotlib)
- Jupyter Notebook
- PCA, KMeans, RandomForest, MLPClassifier

## How to Run

```bash
git clone https://github.com/Utkarsh-Jha-MSAIB/New-York-Crime-With-ML.git
cd New-York-Crime-With-ML
```

