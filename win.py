
# Generate a random dataset with 10 million records and 62 features
import torch
random_inputs = torch.randint(0, 2, (1000, 62), dtype=torch.float)
model = torch.load("model.pth")
model.eval()
# Predict probabilities using the trained model
with torch.no_grad():
    probabilities = model(random_inputs)

# Find the index of the highest probability
max_prob_idx = torch.argmax(probabilities).item()
max_prob = probabilities[max_prob_idx].item()

# Get the ticket (row) with the highest probability
best_ticket = random_inputs[max_prob_idx]

# Find columns where the value is 1
columns_with_1 = [i+1 for i, val in enumerate(best_ticket) if val == 1]

print(f'Highest probability: {max_prob:.6f}')
print(f'Columns with value 1 in the highest probability ticket: {columns_with_1}')
