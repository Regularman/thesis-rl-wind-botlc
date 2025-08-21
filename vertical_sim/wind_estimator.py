import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wind import Wind
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class LSTM_wind_estimator(nn.Module):
  def __init__(self, hidden_dim, input_size):
    super(LSTM_wind_estimator, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_size = input_size
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
    self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, window):
      lstm_out, _ = self.lstm(window)
      velocity_space = self.linear(lstm_out[:,-1,:])
      return velocity_space
    
def prepare_wind(batch):
    wind_strength = np.random.uniform(1,1, size=1).astype(np.float32)
    wind = Wind(wind_strength=wind_strength)
    wind_along = wind.get_along()
    training_data = []
    target = []
    for i in range(len(wind_along)-WINDOW_SIZE):
      interval_start = i 
      interval_end = i + WINDOW_SIZE
      training_data.append(torch.tensor(wind_along[interval_start:interval_end], dtype=torch.float32).unsqueeze(-1))
      target.append(wind_along[i + WINDOW_SIZE])
    '''
    Batching the data
    '''
    if batch:
      training_data = torch.stack(training_data, dim=0)
      target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    return training_data, target

'''
Hyperparameter settings'''
HIDDEN_DIM = 20
LR = 0.05
EPOCH = 100
WINDOW_SIZE = 100
INPUT_SIZE = 1
BATCH_SIZE = 1

def train():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)
  loss_function = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  history = []

  for epoch in tqdm(range(EPOCH)):
    # print(f"Training epoch {i}")
    '''
    Gets the along wind for the estimation. We will take the last 100 time steps of the wind.
    For now, wind strength is set to 25.
    '''
    training_data, target = prepare_wind(batch=True)  

    # Flatten to [num_samples, WINDOW_SIZE, 1]
    # training_data = training_data.squeeze(-1)
    dataset = TensorDataset(training_data, target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    epoch_loss = 0
    for batch_x, batch_y in loader:
      optimizer.zero_grad()
      output = model(batch_x)
      loss = loss_function(output, batch_y)
      loss.backward()
      epoch_loss += loss.item()
      optimizer.step()

    history.append(epoch_loss)

  fig = plt.figure()
  axes = fig.add_subplot(111)
  axes.plot(range(len(history)), history)
  axes.set_title("loss over training epoches")

  torch.save(model.state_dict(), './wind_estimator.mdl')

def eval():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)
  model.load_state_dict(torch.load('./wind_estimator.mdl'))
  # See what the scores are after training
  with torch.no_grad():
    testing_data, target = prepare_wind(batch=False) 
    preds = []
    for data in testing_data:
      data = data.unsqueeze(-1).transpose(0,1)
      preds.append(model(data)[0].item())
      
    error = np.array(preds) - np.array(target)
    
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(range(len(preds)), preds, color="black", alpha=0.3, label="estimated wind")
    axes.plot(range(len(target)), target, color="red", alpha=0.3, label="true wind")
    axes.set_ylabel("Wind velocity magnitude (m/s)")
    axes.set_xlabel("Time (s)")
    axes.set_title("Wind estimator performance")
    axes.legend()

    fig_err = plt.figure()
    axes_err = fig_err.add_subplot(111)
    axes_err.plot(range(len(error)), error, color="black", alpha=0.3)
    axes_err.set_ylabel("Error (m/s)")
    axes_err.set_xlabel("Time (s)")
    axes_err.set_title(f"Wind estimator performance - Error over time - Average error is {np.mean(np.array(error))}")
    plt.show()

def load_wind_estimator():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)
  model.load_state_dict(torch.load('./wind_estimator.mdl'))
  return model, WINDOW_SIZE
if __name__ == "__main__":
   train()
   eval()