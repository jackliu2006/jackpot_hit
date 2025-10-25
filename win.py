
# Generate a random dataset with 10 million records and 62 features
import torch
import torch.nn as nn


# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Generate random inputs with specific constraints
def generate_constrained_inputs(num_samples, num_features):
    inputs = torch.zeros(num_samples, num_features, dtype=torch.float)
    
    for i in range(num_samples):
        # Columns 0-49: exactly 5 columns with value 1
        indices_0_49 = torch.randperm(50)[:5]  # Randomly select 5 indices from 0-49
        inputs[i, indices_0_49] = 1.0
        
        # Columns 50-61: exactly 2 columns with value 1
        indices_50_61 = torch.randperm(12)[:2] + 50  # Randomly select 2 indices from 50-61
        inputs[i, indices_50_61] = 1.0
    
    return inputs

random_inputs = generate_constrained_inputs(2000000, 62)

print(random_inputs[1,:])

model = LogisticRegression(62)
model.load_state_dict(torch.load("model.pth"))
model.eval()
# Predict probabilities using the trained model
with torch.no_grad():
    probabilities = model(random_inputs)

# Find the index of the highest probability
max_prob_idx = torch.argmax(probabilities).item()
max_prob = probabilities[max_prob_idx].item()

# Get the ticket (row) with the highest probability
best_ticket = random_inputs[max_prob_idx]

luck_numbers = [
    (i - 50 + 1) if i > 49 else (i + 1)
    for i, val in enumerate(best_ticket)
    if val == 1
]

print(f'Highest probability: {max_prob:.6f}')
print(f'Columns with value 1 in the highest probability ticket: {luck_numbers}')
