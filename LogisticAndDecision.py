
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Create a folder to store saved models if it doesn't already exist
os.makedirs("saved_models", exist_ok=True)

# Load the balanced and unbalanced diabetes datasets
balanced_data = pd.read_csv("C:/Users/chris/OneDrive/Desktop/Balanced Dataset.csv")
unbalanced_data = pd.read_csv("C:/Users/chris/OneDrive/Desktop/Unbalanced Dataset.csv")

# Add helpful binary features to both datasets based on common health indicators
for data in [unbalanced_data, balanced_data]:
    data['is_elderly'] = (data['Age'] > 60).astype(int)
    data['is_obese'] = (data['BMI'] > 30).astype(int)

# Plot Learning Curves 
def plot_learning_curve(model, title, X, y, cv=5, scoring='f1'):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel(f"{scoring.upper()}")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_mean, 'o-', label="Train", color="r")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation", color="g")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Logistic Regression (No PCA) 
def run_logistic_regression(data, dataset_name):
    print(f"\nLogistic Regression on {dataset_name} (No PCA)")

    # Separate features (X) and target variable (y)
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize feature values using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a baseline Logistic Regression model with default settings
    print("\nBaseline Logistic Regression")
    baseline_lr = LogisticRegression(max_iter=3000)
    baseline_lr.fit(X_train_scaled, y_train)
    baseline_preds = baseline_lr.predict(X_test_scaled)
    print(classification_report(y_test, baseline_preds, target_names=["Non-Diabetic", "Diabetic"]))
    print("Accuracy:", round(accuracy_score(y_test, baseline_preds), 4))

    # Select the top 20 features most correlated with the target variable
    selector = SelectKBest(score_func=f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Use GridSearchCV to tune Logistic Regression hyperparameters and optimize F1 score
    print("\nTuned Logistic Regression (threshold = 0.48)")
    lr_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"]
    }
    tuned_lr = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=3000),
        lr_grid,
        scoring="f1",
        cv=5
    )
    tuned_lr.fit(X_train_selected, y_train)
    best_lr = tuned_lr.best_estimator_

    # Predict probabilities for threshold tuning
    lr_probs = best_lr.predict_proba(X_test_selected)[:, 1]

    # Plot threshold tuning curve to help find a good threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, lr_probs)
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Tuning – {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final threshold choice (manually selected as 0.48 to boost recall)
    final_preds = (lr_probs >= 0.48).astype(int)

    # Show evaluation metrics for the tuned model
    print("Best Parameters:", tuned_lr.best_params_)
    print("CV F1 Score:", round(tuned_lr.best_score_, 4))
    print("CV Recall:", round(cross_val_score(best_lr, X_train_selected, y_train, cv=5, scoring="recall").mean(), 4))
    print(classification_report(y_test, final_preds, target_names=["Non-Diabetic", "Diabetic"]))
    print("Accuracy:", round(accuracy_score(y_test, final_preds), 4))

    # Plot learning curve for Logistic Regression
    plot_learning_curve(best_lr, f"Learning Curve: Logistic Regression ({dataset_name})", X_train_selected, y_train)

    # Save the trained model to disk for later use
    label_clean = dataset_name.replace(" ", "_").lower()
    model_path = f"saved_models/logistic_regression_model_{label_clean}_no_pca.pkl"
    joblib.dump(best_lr, model_path)
    print(f"Model saved to: {model_path}")

# Decision Tree Function 
def run_decision_tree(data, dataset_name):
    print(f"\nDecision Tree on {dataset_name}")

    # Separate features and target variable
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a baseline Decision Tree model
    print("\nBaseline Decision Tree")
    baseline_tree = DecisionTreeClassifier()
    baseline_tree.fit(X_train, y_train)
    baseline_preds = baseline_tree.predict(X_test)
    print(classification_report(y_test, baseline_preds, target_names=["Non-Diabetic", "Diabetic"]))
    print("Accuracy:", round(accuracy_score(y_test, baseline_preds), 4))

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Apply PCA
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    # Use GridSearchCV to tune Decision Tree hyperparameters
    print("\nTuned Decision Tree")
    dt_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 10, 20],
        "criterion": ["gini", "entropy"]
    }
    tuned_decision_tree = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced"),
        dt_grid,
        scoring="f1",
        cv=5
    )
    tuned_decision_tree.fit(X_train_pca, y_train)

    # Evaluate the tuned Decision Tree model
    final_preds = tuned_decision_tree.predict(X_test_pca)
    print("Best Parameters:", tuned_decision_tree.best_params_)
    print("CV F1 Score:", round(tuned_decision_tree.best_score_, 4))
    print("CV Recall:", round(cross_val_score(tuned_decision_tree, X_train_pca, y_train, cv=5, scoring="recall").mean(), 4))
    print(classification_report(y_test, final_preds, target_names=["Non-Diabetic", "Diabetic"]))
    print("Accuracy:", round(accuracy_score(y_test, final_preds), 4))

    # Plot learning curve for Decision Tree
    plot_learning_curve(tuned_decision_tree.best_estimator_, f"Learning Curve: Decision Tree ({dataset_name})", X_train_pca, y_train)

    # Save the tuned decision tree model
    label_clean = dataset_name.replace(" ", "_").lower()
    model_path = f"saved_models/decision_tree_model_{label_clean}.pkl"
    joblib.dump(tuned_decision_tree, model_path)
    print(f"Model saved to: {model_path}")

# Run model training and evaluation pipelines
run_logistic_regression(balanced_data, "Balanced Dataset")
run_decision_tree(balanced_data, "Balanced Dataset")


