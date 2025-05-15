import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Label mapping
mapping = {
    'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0, 'E': 4.0, 'F': 5.0, 'G': 6.0,
    'H': 7.0, 'I': 8.0, 'J': 9.0, 'K': 10.0, 'L': 11.0, 'M': 12.0, 'N': 13.0,
    'O': 14.0, 'P': 15.0, 'Q': 16.0, 'R': 17.0, 'S': 18.0, 'T': 19.0,
    'U': 20.0, 'V': 21.0, 'W': 22.0, 'X': 23.0, 'Y': 24.0, 'Z': 25.0,
    'del': 26.0, 'space': 27.0
}
inv_mapping = {v: k for k, v in mapping.items()}

# Load and preprocess data
df = pd.read_csv("result_cleaned.csv")
df['label'] = df['label'].map(mapping)
x = df.drop('label', axis=1).values
y = df['label'].values.astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=41)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

# Define the model
class Model(nn.Module):
    def __init__(self, input_features=10, h1=50, h2=50, output=28):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Load trained model
model = Model()
model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu')))
model.eval()

# Evaluate loss
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    y_pred_logits = model(x_test)
    test_loss = criterion(y_pred_logits, y_test)

# Convert to numpy
y_pred = torch.argmax(y_pred_logits, dim=1).numpy()
y_true = y_test.numpy()

# Accuracy
acc = accuracy_score(y_true, y_pred)

# Report and Confusion Matrix
report = classification_report(y_true, y_pred, target_names=[inv_mapping[i] for i in range(28)])
cm = confusion_matrix(y_true, y_pred)

# Print results
print(f"✅ Test Loss: {test_loss.item():.4f}")
print(f"✅ Accuracy: {acc * 100:.2f}%\n")
print("✅ Classification Report:")
print(report)
print("✅ Confusion Matrix:")
print(cm)
