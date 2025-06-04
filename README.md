# Diabetes-prediction
# 🩺 Diabetes Prediction Using Machine Learning

This project predicts whether a person is likely to have diabetes based on health data using a machine learning model trained on the Pima Indians Diabetes dataset.

---

## 📊 Dataset

The dataset used is **`diabetes.csv`**, containing 768 entries and 8 features related to health metrics:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target variable is `Outcome` (0 = No Diabetes, 1 = Diabetes).

---

## 🧠 Model Used

A **K-Nearest Neighbors (KNN)** classifier is used for training the model, along with data scaling using `StandardScaler`.

Model training was done in a Jupyter Notebook, and the final trained model was exported as:

- `diabetes_model.pkl`
- `scaler.pkl`

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
