# =====================================================
# Project Title : Hotel Reservation System using KNN
# Author        : Atharv Nikam
# Description   : This project predicts whether a hotel
#                 reservation will be canceled or not
#                 using K-Nearest Neighbors classifier.
# =====================================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "hotel_registration.csv")


if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset file not found. Check the file path.")

data = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(data.head(), "\n")

# -------------------------------
# STEP 2: Check Missing Values
# -------------------------------
print("Missing Values:")
print(data.isnull().sum(), "\n")

# -------------------------------
# STEP 3: Encode Categorical Data
# -------------------------------
categorical_columns = [
    'type_of_meal_plan',
    'room_type_reserved',
    'market_segment_type',
    'booking_status'
]

encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

print("Categorical columns encoded successfully\n")

# -------------------------------
# STEP 4: Feature & Target Split
# -------------------------------
X = data.drop(columns=['booking_status', 'Booking_ID'])
y = data['booking_status']

# -------------------------------
# STEP 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -------------------------------
# STEP 6: Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# STEP 7: Train KNN Model
# -------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

print("KNN Model trained successfully\n")

# -------------------------------
# STEP 8: Model Evaluation
# -------------------------------
y_pred = knn_model.predict(X_test_scaled)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 9: Sample Prediction
# -------------------------------
sample_data = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample_data)
prediction = knn_model.predict(sample_scaled)

status = "Canceled" if prediction[0] == 1 else "Not Canceled"
print("\nSample Booking Prediction:", status)
