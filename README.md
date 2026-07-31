# Customer Churn Prediction

An end-to-end machine learning project that predicts whether an e-commerce customer will churn, built using a classification pipeline with a live Streamlit UI and a FastAPI endpoint.

---

## Problem Statement

Customer churn is one of the most costly problems in e-commerce. This project builds a binary classification model that identifies high-risk customers before they leave, enabling businesses to act early with targeted retention strategies.

---

## Demo

| | Link |
|---|---|
| Streamlit App | `https//churn-predictor-ui.onrender.com` |
| FastAPI Docs | `https//churn-predictor-api.onrender.com/docs` |

---

## Dataset

- **Source:** E-commerce Customer Churn Dataset (Kaggle)
- **Size:** 3,941 rows, 11 features
- **Churn rate:** 16.3% (imbalanced)
- **Key features:** Tenure, Complain, CashbackAmount, PreferedOrderCat, MaritalStatus

The dataset was validated before use — a synthetically generated alternative was rejected due to near-perfect CustomerID leakage and an artificially balanced 52/47 churn split.

---

## Project Structure

```
churn-prediction/
├── data/
│   └── data_ecommerce_customer_churn.csv
├── model/
│   ├── xgboost_churn_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── Churn_Prediction_EDA.ipynb
│   └── Churn_Prediction_Feature-Engineering__Modelling.ipynb
├── app.py
├── api.py
├── requirements.txt
├── runtime.txt
├── LICENSE
└── README.md
```

---

## Pipeline

**Data Cleaning**
- Removed 671 fully duplicate rows
- Imputed nulls in Tenure, WarehouseToHome, DaySinceLastOrder with column medians
- Merged redundant category labels: `Mobile Phone` → `Mobile`

**EDA Findings**
- `Tenure` is the strongest predictor — new customers churn significantly more
- `Complain` customers churn at nearly 3× the base rate
- Mobile category customers churn at 27% — 6.5× higher than Grocery (4%)
- Single customers churn at 26% vs Married at 11%

**Feature Engineering**
- One-hot encoded `PreferedOrderCat` and `MaritalStatus`
- Log-transformed skewed columns: Tenure, WarehouseToHome, DaySinceLastOrder, CashbackAmount

**Modelling**
- Stratified 80/20 train/test split
- StandardScaler fitted on train only, applied to test
- SMOTE applied on training set only to handle 83/17 class imbalance
- Trained and compared 5 classifiers: Logistic Regression, KNN, SVM, Random Forest, XGBoost

**Model Comparison (Churn class — label `1`)**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.48 | 0.92 | 0.63 | 0.922 |
| KNN | 0.51 | 0.90 | 0.65 | 0.910 |
| SVM | 0.54 | 0.84 | 0.66 | 0.919 |
| Random Forest | 0.73 | 0.78 | 0.75 | 0.956 |
| **XGBoost** | **0.78** | **0.79** | **0.78** | **0.957** |

XGBoost selected as the final model — best precision/recall balance and highest ROC-AUC.

**Hyperparameter Tuning**
GridSearchCV with StratifiedKFold (5 folds), optimising for F1. Tuned model (max_depth=7, n_estimators=300) achieved CV F1 of 0.954 but test F1 of 0.77 — slightly below the untuned baseline. Untuned XGBoost retained as the final model, demonstrating that CV score alone is insufficient proof of generalisation on real imbalanced data.

**Explainability**
SHAP TreeExplainer used for global feature importance. Top drivers:
1. Tenure — low tenure strongly increases churn risk
2. Complain — complaints are the strongest positive churn signal
3. NumberOfAddress — hidden interaction effects not visible in correlation alone
4. CashbackAmount — higher cashback retains customers

Notable: `PreferedOrderCat_Mobile` ranked 13th in SHAP despite having the strongest raw correlation (+0.22). Its effect is already captured by Tenure and Complain — illustrating the difference between univariate correlation and model-level feature importance.

---

## Deployment

Both services deployed independently on Render, each loading the saved model directly.

**Run Streamlit locally:**
```bash
streamlit run app.py
```

**Run FastAPI locally:**
```bash
uvicorn api:app --reload
```

> Note: Free tier on Render spins down after 15 minutes of inactivity. First request after idle may take ~30 seconds to wake up.

---

## Tech Stack

- **Language:** Python 3.x
- **Modelling:** scikit-learn, XGBoost, imbalanced-learn
- **Explainability:** SHAP
- **UI:** Streamlit
- **API:** FastAPI, Uvicorn
- **Deployment:** Render

---

## Author

**Samarth Singh**
- GitHub: [OfficialSamarthsingh](https://github.com/OfficialSamarthsingh)
- LinkedIn: [samarthsinghofficial](https://linkedin.com/in/samarthsinghofficial)
- Portfolio: [samarthsinghportfolio.netlify.app](https://samarthsinghportfolio.netlify.app)
