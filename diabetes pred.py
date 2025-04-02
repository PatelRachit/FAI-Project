
# Christeena Jacob
# Checking how well Logistic Regression and Decision Tree perform on diabetes data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
unbalanced_data = pd.read_csv("C:/Users/chris/OneDrive/Desktop/Diabetes data1.csv")
balanced_data = pd.read_csv("C:/Users/chris/OneDrive/Desktop/Diabetes data2.csv")


# This function checks model performance
def model_performance(data, label):
    print(f"\n--- Model Performance on {label} ---")

    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    logistic_preds = logistic_model.predict(X_test_scaled)
    print("\nLogistic Regression:")
    print("Accuracy:", accuracy_score(y_test, logistic_preds))
    print(classification_report(y_test, logistic_preds))

    # Decision Tree
    decision_model = DecisionTreeClassifier()
    decision_model.fit(X_train, y_train)
    decision_preds = decision_model.predict(X_test)
    print("Decision Tree:")
    print("Accuracy:", accuracy_score(y_test, decision_preds))
    print(classification_report(y_test, decision_preds))

# Run the function on both datasets
model_performance(unbalanced_data, "Unbalanced Dataset")
model_performance(balanced_data, "Balanced Dataset")
