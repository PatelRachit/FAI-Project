import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X, "\n")
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# print(X_train)
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html 
# It seems that there is no big differences between relu/tanh, or between optimizer sgd/adam. ?increasing max_interation only increase the accuracy slowly. 
# Changing hidden_layer_size to 128,32 does not make a big differnce also.
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='sgd', max_iter=5000, learning_rate='adaptive')
model.fit(X_train, y_train)


columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
new_data = pd.DataFrame([[
    0.0,1.0,1.0,37.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,4.0,0.0,0.0,0.0,0.0,6.0,4.0,1.0
]], columns=columns)
X_test_scaled = scaler.transform(X_test)
new_input_scaled = scaler.transform(new_data)

result = model.predict(new_input_scaled)
print(result, "\n")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred), "\n")
print("\nClassification Report:\n", classification_report(y_test, y_pred))