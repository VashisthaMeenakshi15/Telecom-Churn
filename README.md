# Telco Customer Churn Prediction

## a. Problem Statement
The telecommunications industry faces a significant challenge with customer attrition, also known as **churn**. 
Acquiring a new customer is estimated to be 5-25 times more expensive than retaining an existing one. 

The objective of this project is to develop a Machine Learning solution that predicts which customers are at high risk of leaving the service. By analyzing historical customer data—such as demographics, service usage, and billing information—we treat this as a **Binary Classification** problem (Churn vs. No Churn). This predictive capability allows the business to proactively intervene with retention strategies (e.g., discounts, targeted offers) to reduce revenue loss.

## b. Dataset Description
The model was trained on the **Telco Customer Churn** dataset.
* **Target Variable:** `Churn` (Yes/No)
* **Dataset Size:** Approximately 7,043 rows and 21 columns.
* **Key Features:**
    * **Demographics:** Gender, Senior Citizen status, Partner, Dependents.
    * **Services:** Phone Service, Multiple Lines, Internet Service (DSL, Fiber Optic), Online Security, Tech Support, Streaming TV/Movies.
    * **Account Information:** Tenure (months a customer has stayed), Contract Type (Month-to-month, One year, Two year), Payment Method, Monthly Charges, and Total Charges.

## c. Models Used
We implemented six different machine learning algorithms to compare performance. Below is the evaluation metric comparison for each model on the test dataset.

### Comparison Table
| ML Model Name               | Accuracy |   AUC | Precision | Recall |    F1 |   MCC |
| :-------------------------- | :------- | :---- | :-------- | :----- | :---- | :---- |
| **Logistic Regression** | 0.816    | 0.858 | 0.680     | 0.587  | 0.630 | 0.517 |
| **Decision Tree** | 0.725    | 0.655 | 0.489     | 0.505  | 0.497 | 0.366 |
| **kNN** | 0.768    | 0.776 | 0.565     | 0.535  | 0.549 | 0.435 |
| **Naive Bayes** | 0.743    | 0.826 | 0.514     | 0.791  | 0.623 | 0.472 |
| **Random Forest (Ensemble)**| 0.793    | 0.835 | 0.635     | 0.492  | 0.554 | 0.449 |
| **XGBoost (Ensemble)** | **0.806**| **0.849** | **0.659** | **0.535** | **0.590** | **0.494** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Strong Baseline.** It achieved the highest accuracy (0.816) and a very competitive AUC (0.858). This suggests that the relationship between features (like tenure and charges) and churn is largely linear. It is highly interpretable but may struggle with very complex customer behaviors. |
| **Decision Tree** | **Weakest Performer.** It had the lowest accuracy (0.725) and AUC (0.655). The low MCC score (0.366) indicates it performed only slightly better than random guessing in some cases. It clearly suffered from overfitting to the training data. |
| **kNN** | **Average Performance.** With an accuracy of 76.8%, it performed decently but struggled with the high dimensionality of the dataset. It is computationally expensive during inference and didn't offer any significant advantage over linear models here. |
| **Naive Bayes** | **High Recall Specialist.** While its accuracy was lower (0.743), it achieved the highest Recall (0.791) of all models. This means it is excellent at catching churners, but it raises many false alarms (low precision of 0.514) due to its assumption that features are independent. |
| **Random Forest (Ensemble)** | **Stable but Conservative.** It performed well (Accuracy 0.793) but had a lower recall (0.492) compared to others. It successfully reduced the variance of the single decision tree but was surprisingly outperformed by the simpler Logistic Regression on this specific split. |
| **XGBoost (Ensemble)** | **Balanced & Robust.** It achieved the second-best accuracy (0.806) and a high AUC (0.849). It provided a good balance between Precision and Recall (F1: 0.590), making it a reliable choice for deployment where we need to balance catching churners with avoiding false positives. |

### Project Structure
```text
project-folder/
├── app.py                     # Streamlit App (Frontend)
├── ml_assignment_2_streamlit.py   # Training Pipeline (Backend)
├── requirements.txt           # Required libraries
├── README.md                  # Project Document
├── model_performance.csv      # Metrics
└── model/                     # Serialized Model Files
    ├── log_reg.pkl
    ├── dt_clf.pkl
    ├── knn.pkl
    ├── gnb.pkl
    ├── rf_clf.pkl
    └── xgb.pkl

### Live Application
Streamlit application access link to test the model predictions in real-time: [Telecom churn](https://telecom-churn-kbubljzpjsgkjdee4jetbz.streamlit.app/)
