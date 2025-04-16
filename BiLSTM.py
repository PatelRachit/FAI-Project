import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import RandomOverSampler

# Load data
data = pd.read_csv("dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
X = data.drop("Diabetes_binary", axis=1).values.astype(np.float32)
y = data["Diabetes_binary"].values.astype(np.float32)

# Normalize and use random oversampling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
X_resampled = X_resampled.reshape(X_resampled.shape[0], 1, X_resampled.shape[1]) # reshape for LSTM

# Split and transfer data into tensor
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Define Bidirectional - LSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

# Initialize model
model = BiLSTM(input_size=X_train.shape[2])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()
    predicted_labels = (predictions >= 0.4).astype(int)

print("Accuracy:", accuracy_score(y_test, predicted_labels), "\n")
print("\nClassification Report:\n", classification_report(y_test, predicted_labels))
