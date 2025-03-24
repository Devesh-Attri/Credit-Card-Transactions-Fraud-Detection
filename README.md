# Fraud Detection Model

## 📌 Overview
This project implements a **fraud detection system** using **machine learning** to classify transactions as **fraudulent or legitimate**. The model is trained on a dataset containing transaction details such as amount, user ID, merchant information, timestamps, and other relevant attributes.

## 🚀 Features
- **Preprocessing**: Extracts time-based features, encodes categorical variables, and computes user age from date of birth.
- **Feature Engineering**: Uses transaction amount, location, time, user demographics, and merchant category.
- **Model Training**: Uses a **Random Forest Classifier** to detect fraudulent transactions.
- **Performance Metrics**: Evaluates model accuracy, precision, recall, and F1-score.
- **Misclassification Analysis**: Identifies misclassified transactions for further investigation.

## 💾 Download the Dataset
You can download the dataset from Kaggle using the below given link:
```bash
kaggle.com/datasets/kartik2112/fraud-detection?resource=download
```

## 🗂 Dataset Description
The dataset contains the following columns:
- `trans_date_trans_time`: Timestamp of the transaction.
- `cc_num`: Credit card number (hashed for anonymity).
- `merchant`: Merchant where the transaction took place.
- `category`: Type of merchant.
- `amt`: Transaction amount.
- `gender`: Gender of the cardholder.
- `city_pop`: Population of the transaction location.
- `job`: Cardholder’s occupation.
- `dob`: Date of birth (used to calculate age).
- `is_fraud`: Target variable (1 = Fraud, 0 = Legitimate).

## 🔧 Installation
To run this project, install the required dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## 📊 Model Training & Evaluation
### **1️⃣ Load and Preprocess Data**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### **2️⃣ Feature Engineering**
```python
# Convert transaction time to datetime
df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
df_train['hour'] = df_train['trans_date_trans_time'].dt.hour
df_train['age'] = df_train['dob'].apply(lambda x: 2025 - int(x[:4]))
```

### **3️⃣ Train the Model**
```python
X_train, X_val, y_train, y_val = train_test_split(df_train[features], df_train['is_fraud'], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### **4️⃣ Evaluate Performance**
```python
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
```

## 📈 Model Performance
### **Validation Results**
- **Accuracy**: `99.84%`
- **Precision (Fraudulent Class)**: `0.96`
- **Recall (Fraudulent Class)**: `0.76`

### **Test Results**
- **Accuracy**: `99.86%`
- **Precision (Fraudulent Class)**: `0.94`
- **Recall (Fraudulent Class)**: `0.70`

## 💾 Save and Load Model
To save the trained model:
```python
import joblib
joblib.dump(model, 'fraud_detection_model.pkl')
```
To load and use the model:
```python
loaded_model = joblib.load('fraud_detection_model.pkl')
predictions = loaded_model.predict(X_test)
```

## 📌 Conclusion
This fraud detection model efficiently identifies fraudulent transactions while keeping **false positives low**. Further improvements can be made using **deep learning** or **anomaly detection techniques**.

📢 **Feel free to contribute or suggest enhancements!** 🚀"""
