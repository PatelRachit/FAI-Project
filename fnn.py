import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv("dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
param_dist = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64, 32)], 
    'activation': ['logistic', 'relu', 'tanh', ],
    'solver': ['adam', 'sgd', ],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [20000],
    'early_stopping': [True],
    'n_iter_no_change': [20, 40]
}

X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X, "\n")
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# print(X_train)
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html 
# It seems that there is no big differences between relu/tanh, or between optimizer sgd/adam. ?increasing max_interation only increase the accuracy slowly. 
# Changing hidden_layer_size to 128,32 does not make a big differnce also.
# model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=10000, n_iter_no_change=20,learning_rate='adaptive') - very poor performance
model = MLPClassifier(random_state=42)
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=30,  
    scoring='recall', 
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)
print("Best parameters found:\n", search.best_params_)

# Evaluate
y_pred = search.best_estimator_.predict(X_test)

columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
new_data = pd.DataFrame([[
    0.0,1.0,1.0,37.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,4.0,0.0,0.0,0.0,0.0,6.0,4.0,1.0
]], columns=columns)

new_input_scaled = scaler.transform(new_data)
result = search.best_estimator_.predict(new_input_scaled)
print(result, "\n")

print("Accuracy:", accuracy_score(y_test, y_pred), "\n")
print("\nClassification Report:\n", classification_report(y_test, y_pred))