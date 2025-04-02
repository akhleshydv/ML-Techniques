# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load the Diabetes Dataset
diabetes = load_diabetes()
X = diabetes.data   # Features
y = (diabetes.target > diabetes.target.mean()).astype(int)  # Convert target to binary (1 if above mean, else 0)

# Convert dataset into DataFrame
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

# Save to CSV (optional)
df.to_csv('diabetes_data.csv', index=False)

# Display first few rows
print("🔹 First 5 Rows of the Dataset:")
print(df.head())

# 2️⃣ Exploratory Data Analysis (EDA)
print("\n🔹 Dataset Information:")
print(df.info())

print("\n🔹 Dataset Description:")
print(df.describe())

# Plot target distribution
sns.countplot(x=df['target'])
plt.title("Distribution of Target Classes (0: Low, 1: High)")
plt.show()

# 3️⃣ Data Preprocessing: Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n🔹 Training Data Shape:", X_train.shape)
print("🔹 Testing Data Shape:", X_test.shape)

# 4️⃣ Building the Naïve Bayes Classifier
model = GaussianNB()

# Train the Model
model.fit(X_train, y_train)

# 5️⃣ Predicting on Test Data
y_pred = model.predict(X_test)

# 6️⃣ Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("\n🔹 Model Accuracy:", accuracy)

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Diabetes', 'High Diabetes']))

print("\n🔹 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Low Diabetes', 'High Diabetes'], yticklabels=['Low Diabetes', 'High Diabetes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
