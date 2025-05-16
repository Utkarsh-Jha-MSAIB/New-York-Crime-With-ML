
# 🔍 NYC Neighborhood Safety Classification Using Machine Learning (2010–2024)

## 📄 Abstract
This project explores whether machine learning can accurately classify neighborhood safety in New York City using arrest and shooting data from 2010 to 2024. After applying **PCA** and **K-Means clustering** to define safety labels on the arrest data, these safety labels were then transferred into the shooting data at the same granularity. Two supervised models — **Random Forest** and **Artificial Neural Network (ANN)** — were trained to predict one of two labels:
- **Unsafe and Vulnerable to Shooting**
- **Safe but Vulnerable to Shooting**

Results showed **high predictive accuracy** and revealed key predictors such as jurisdiction code, time of day, holiday indicators, and proximity to major infrastructure. These findings offer **actionable insights for data-informed policy** aimed at improving public safety in NYC.

## 📌 Introduction
As crime patterns in NYC evolve, particularly in the post-2019 context, this project investigates whether machine learning can reliably classify neighborhoods as unsafe using historical arrest and shooting data. Understanding **where and why shootings occur** is critical for public safety and resource allocation.

This analysis addresses the limitations of one-size-fits-all policies by using data to uncover **neighborhood-specific vulnerabilities**, offering insights to guide targeted interventions.

## 📁 Repository Structure

```
<pre> ├── Crime Data Clustering.ipynb # Clustering using PCA + K-Means ├── Shooting Data Processing.ipynb # Feature engineering & preprocessing ├── Shooting Data Prediction.ipynb # Model training & evaluation ├── src/ │ ├── Data_Preprocessing.py # Custom preprocessing functions │ └── ML_Utilities.py # Utility functions for modeling ├── data/ │ ├── raw/ # Original files (Excel, Word) │ │ ├── Arrest Data Link.docx │ │ └── NYPD_Shooting_Incident_Data_Historic.xlsx │ └── processed/ # Cleaned & engineered datasets │ ├── Crime_Data_Clustered.xlsx │ ├── Shooting_Data_Processed.xlsx │ └── Shooting_Data_For_Prediction.xlsx ├── output/ │ ├── ann/ # ANN model results │ │ ├── ann_grid_results.csv │ │ ├── confusion_matrix.png │ │ └── feature_importance.png │ ├── rf/ # Random Forest model results │ │ ├── rf_grid_results.csv │ │ ├── confusion_matrix.png │ │ └── feature_importance.png │ └── insights/ │ └── Key Visualizations.pdf ├── .gitignore └── README.md # Project overview and setup </pre>```

## ⚙️ Model Summary

| Model         | Accuracy | Notes                                      |
|---------------|----------|--------------------------------------------|
| ANN (MLPClassifier) | ~85%     | Captures non-linear patterns well       |
| Random Forest       | **91%**  | Most interpretable; top performer       |

## 📊 Evaluation

### 🔹 Confusion Matrix – ANN
- 799 true safe, 1291 true unsafe
- Moderate false positives

### 🔹 Confusion Matrix – Random Forest
- Higher true positive/negative count
- Fewer false predictions

## 🌟 Feature Importance

Key predictors in both models:
- `JURISDICTION_CODE`
- `Times Square Distance`
- `After_6PM_Flag`
- Holiday flags (e.g., Thanksgiving, Independence Day)
- `Murder_Flag`, `Unemployment_Flag`, seasonality

## 🧠 Insights & Impact
- Spatial and temporal factors significantly influence shooting vulnerability.
- RF provided more interpretable results, ANN was more balanced.
- Findings highlight neighborhoods that are **"Safe but Vulnerable"**, a critical insight for preemptive policing and community interventions.

## 🛠️ Tools & Libraries
- Python (Scikit-learn, Pandas, NumPy, Matplotlib)
- Jupyter Notebook
- PCA, KMeans, RandomForest, MLPClassifier

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/nyc-crime-safety-ml.git
cd nyc-crime-safety-ml
```
Then run the notebooks in this order:
1. `Shooting Data Processing.ipynb`
2. `Crime Data Clustering.ipynb`
3. `Shooting Data Prediction.ipynb`

## 📬 Contact

For questions, collaboration, or feedback, please open an issue or contact via GitHub.

---
**Disclaimer**: Data used in this project is public and anonymized. Interpretations are meant for educational and analytical purposes only.
