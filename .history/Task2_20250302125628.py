import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Generate sine wave dataset
def generate_sine_data(seq_length=30, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    y = np.ones_like(x)
    sequences = []
    for i in range(len(y) - seq_length):
        sequences.append(y[i : i + seq_length])
    return np.array(sequences)

# Parameters
seq_length = 30
input_size = 1
num_epochs = 10
batch_size = 16
hidden_dim = 64
num_layers = 4
learning_rate = 0.001

# Prepare dataset
data = generate_sine_data(seq_length)
x_train = torch.tensor(data[:, :20], dtype=torch.float32).unsqueeze(-1)  # Last 20 points as input
y_train = torch.tensor(data[:, 20:], dtype=torch.float32).unsqueeze(-1)  # Next 10 points as target
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.embedding = nn.Linear(input_size, hidden_dim)

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch, seq_len, hidden_dim]
        x = self.transformer(x)  # Shape: [batch, seq_len, hidden_dim]
        x = self.fc(x[:, -1, :])  # Take the last time step
        return x.unsqueeze(-1)  # Match target shape [batch_size, 10, 1]


# Model, loss, optimizer
model = TransformerModel(input_size, hidden_dim, num_layers, output_size=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Test prediction
with torch.no_grad():
    test_input = x_train[0].unsqueeze(0)  # Take the first sequence for testing
    prediction = model(test_input).squeeze().numpy()
    ground_truth = y_train[0].squeeze().numpy()
    
    plt.plot(range(20), test_input.squeeze().numpy(), label='Input')
    plt.plot(range(20, 30), ground_truth, label='Ground Truth')
    plt.plot(range(20, 30), prediction, label='Prediction', linestyle='dashed')
    plt.legend()
    plt.show()
