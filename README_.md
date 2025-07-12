
# 🌍 Kiva + World Bank Loan Prediction Project

This project analyzes and predicts both the **likelihood** and **amount** of loan funding on [Kiva.org](https://www.kiva.org) by integrating Kiva loan data with World Bank development indicators. It includes comprehensive exploratory data analysis (EDA), outlier detection and treatment, feature engineering, preprocessing pipelines, and the application of multiple machine learning algorithms for both **classification and regression** tasks.

---

## 📁 Project Structure

```
project_kiva_worldbank/
│
├── config/                     # Configuration files (e.g., config.yaml)
├── data/                      # Raw and processed datasets
│   ├── raw/                   # Original input CSVs
│   └── processed/             # Cleaned, merged, and featured datasets
│
├── model/                     # Trained model files (.pkl)
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── logistic_regression_smote_impute.pkl
│   └── funded_amount_regression.pkl
│
├── notebooks/                 # Jupyter notebooks for EDA and modeling
│   ├── EDA_kiva_worldbank.ipynb
│   └── modeling.ipynb
│
├── outputs/                   # Outputs generated from the project
│   ├── PLOTS/                 # Plots and visuals
│   │   ├── PLOTS.pdf
│   │   └── funded_amount_regression_scatter.png
│   └── report/                # Evaluation summaries and reports
│       ├── logistic_regression_evaluation.txt
│       ├── logistic_regression_summary.txt
│       ├── logistic_regression_smote_summary.txt
│       ├── random_forest_summary.txt
│       ├── xgboost_summary.txt
│       ├── model_comparison_summary.txt
│       └── funded_amount_regression_summary.txt
│
├── scripts/                   # Modular Python scripts for reproducibility
│   ├── preprocessing.py
│   ├── model_utils.py
│   ├── train_multiple_models.py
│   └── visualization.py
│
├── requirements.txt           # Python dependencies
└── README.md                  # summary of project
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Models Used (Classification & Regression)

- Logistic Regression
- Logistic Regression with SMOTE & Imputation
- Random Forest
- XGBoost
- Random Forest Regressor (for predicting funded amount)

All models are saved in `model/` as `.pkl` files and can be loaded using `joblib`.

---

## 📈 Project Highlights

- ✔️ EDA on loan and development data
- ✔️ Merging datasets from different sources
- ✔️ Outlier detection using IQR and visualized with boxplots
- ✔️ SMOTE to handle class imbalance
- ✔️ Pipeline with imputation, scaling, and modeling
- ✔️ Evaluation using accuracy, precision, recall, f1-score, confusion matrix
- ✔️ Cross-validation and model comparison
- ✔️ Regression pipeline for predicting loan funded amount

---

## 📦 How to Use

1. Clone the repository  
2. Install the requirements  
3. Run `EDA_kiva_worldbank.ipynb` for data exploration  
4. Run `modeling.ipynb` or `train_multiple_models.py` for model training  
5. Explore outputs in the `outputs/` folder  

---

## 📁 Outputs

- 📈 **PLOTS.pdf** — Boxplots and visuals for cleaned features
- 📊 **Evaluation Reports** — Found in `outputs/report/`
- 🧠 **Trained Models** — `.pkl` files under `model/`
- 📉 **Regression Summary** — `funded_amount_regression_summary.txt`

---

## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Model deployment with Streamlit or Flask
- Evaluate multiple regression algorithms (e.g., Lasso, Gradient Boosting Regressor)
- Export results to a dashboard

---

## 🙌 Credits

This project was built using public data from:
- [Kiva.org](https://www.kiva.org)
- [World Bank Open Data](https://data.worldbank.org)

---

## 🧠 Author

**ADISHI AGRAWAL**  
Feel free to connect via GitHub or LinkedIn.
