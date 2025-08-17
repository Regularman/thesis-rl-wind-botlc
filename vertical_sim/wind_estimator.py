import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wind import Wind
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import numpy as np

class LSTM_wind_estimator(nn.Module):
  def __init__(self, hidden_dim, input_size):
    super(LSTM_wind_estimator, self).__init__()
    self.hidden_dim = hidden_dim
    self.input_size = input_size
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim)
    self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, window):
      lstm_out, _ = self.lstm(window)
      velocity_space = self.linear(lstm_out[-1])
      return velocity_space
    
def prepare_wind(batch):
    wind_strength = 25
    wind = Wind(wind_strength=wind_strength)
    wind_along = wind.get_along()
    training_data = []
    target = []
    for i in range(len(wind_along)-WINDOW_SIZE):
      interval_start = i 
      interval_end = i + WINDOW_SIZE
      training_data.append(torch.tensor(wind_along[interval_start:interval_end], dtype=torch.float32).unsqueeze(1).unsqueeze(1))
      target.append(wind_along[i + WINDOW_SIZE])

    '''
    Batching the data
    '''
    if batch:
      training_data = torch.cat(training_data, dim=1)
      target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    return training_data, target

'''
Hyperparameter settings'''
HIDDEN_DIM = 10
LR = 0.1
EPOCH = 100
WINDOW_SIZE = 100
INPUT_SIZE = 1

def train():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)
  loss_function = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  history = []

  for i, epoch in enumerate(range(EPOCH)):
    print(f"Training epoch {i}")
    '''
    Gets the along wind for the estimation. We will take the last 100 time steps of the wind.
    For now, wind strength is set to 25.
    '''
    training_data, target = prepare_wind(batch=True)

    model.zero_grad()

    predicted_velocity = model(training_data)

    loss = loss_function(predicted_velocity, target)

    loss.backward()
    optimizer.step()
    print(f"Epoch {i} loss: {loss.item()}")
    history.append(loss.item())

  torch.save(model.state_dict(), './wind_estimator.mdl')

def eval():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)
  model.load_state_dict(torch.load('./wind_estimator.mdl'))
  # See what the scores are after training
  with torch.no_grad():
    testing_data, target = prepare_wind(batch=False) 
    preds = []
    for data in testing_data:
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

if __name__ == "__main__":
  #  train()
   eval()