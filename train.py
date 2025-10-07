import os
import pandas as pd


df=pd.read_csv("jackpot.csv",sep=';',engine='python', header=None).fillna(0)
print(df.shape) 
zero = df.columns[(df==0).all()] 
df=df.drop(columns=zero, axis=1) #delete empty column

df_win=df.iloc[:,2:] # remove first two columns
df_win=df_win.map(lambda x: 1 if x != 0 else 0)
cols = list(range(1,51))
cols.extend(list(range(1,13)))

print(df_win.shape)
df_win.columns = cols
print(df_win)



import torch
win_inputs = torch.tensor(df_win.values, dtype=torch.int)
label_col = torch.ones(win_inputs.shape[0],1, dtype=torch.int)
win_tensor = torch.cat((win_inputs,label_col),dim=1)

fake_inputs_shape = (5000000,62)
fake_inputs = torch.randint(0,2,fake_inputs_shape,dtype=torch.int)
fake_label_col = torch.zeros(fake_inputs.shape[0],1, dtype=torch.int)
fake_tensor = torch.cat((fake_inputs,fake_label_col),dim=1)

traning_data = torch.cat((win_tensor,fake_tensor),dim=0)

import torch.nn as nn
import torch.optim as optim

# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Prepare data
X = traning_data[:, :-1].float()
y = traning_data[:, -1].float().unsqueeze(1)


MODEL_PATH = "model.pth"
# if os.path.exists(MODEL_PATH):
#     model = LogisticRegression(X.shape[1])
#     model.load_state_dict(torch.load(MODEL_PATH))
# else:
model = LogisticRegression(X.shape[1])
model.train()


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# save model for next time
torch.save(model.state_dict(), "model.pth")