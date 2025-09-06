import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from wind import Wind
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from vertical_sim import eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

'''
Hyperparameter settings
'''
WINDOW_SIZE = 100
INPUT_SIZE = 8
HIDDEN_DIM = 32
LR = 0.05
EPOCH = 600
BATCH_SIZE = 4
NUM_LAYERS = 2
DROPOUT = 0.1

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
    '''
    Obtain the wind and action data 
    '''
    error, failed, out_of_bounds, wind, bearing, v_x, v_y, omega, pitch, action_left, action_right = eval(render=False)
    wind_along = wind[:, 0]
    wind_along_target = wind_along[WINDOW_SIZE:]

    ## Wind vertical is currently set to be 0
    wind_vertical = wind[:, 1]
    wind_vertical_target = wind_vertical[WINDOW_SIZE:]

    target = torch.tensor(wind_along_target)
    # target = torch.stack([torch.tensor(wind_along_target, dtype=torch.float32), 
    #                       torch.tensor(wind_vertical_target, dtype=torch.float32)], dim=-1)

    '''
    Inputs to the LSTM. Normalised.
    '''
    windows = []
    training_data = torch.stack([torch.tensor(error, dtype=torch.float32), 
                                 torch.tensor(bearing, dtype=torch.float32), 
                                 torch.tensor(omega, dtype=torch.float32), 
                                 torch.tensor(pitch, dtype=torch.float32), 
                                 torch.tensor(v_x, dtype=torch.float32), 
                                 torch.tensor(v_y, dtype=torch.float32),
                                 torch.tensor(action_left, dtype=torch.float32),
                                 torch.tensor(action_right, dtype=torch.float32)], dim=-1)
    
    for i in range(len(training_data) - WINDOW_SIZE):
        windows.append(training_data[i:i+WINDOW_SIZE])
    training_data = torch.stack(windows)

    return training_data, target

def train():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE).to(device)
  loss_function = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  history = []

  for epoch in tqdm(range(EPOCH)):
    '''
    Gets the along wind for the estimation. We will take the last 100 time steps of the wind.
    For now, wind strength is set to be uniformly distributed between a wind strength of 0-25. 
    '''
    distance, failed, out_of_bounds, wind, bearing, v_x, v_y, omega, pitch, action_left, action_right = eval(render=False)
    # training_data, target = prepare_wind(batch=True)  
    distance = torch.tensor(distance, dtype=torch.float32).to(device)
    bearing = torch.tensor(bearing, dtype=torch.float32).to(device)
    omega = torch.tensor(omega, dtype=torch.float32).to(device)
    pitch = torch.tensor(pitch, dtype=torch.float32).to(device)
    v_x = torch.tensor(np.array(v_x), dtype=torch.float32).to(device)
    v_y = torch.tensor(np.array(v_y), dtype=torch.float32).to(device)
    wind_along =  torch.tensor((wind[:, 0]), dtype=torch.float32).to(device)   
    wind_vertical =  torch.tensor(wind[:, 1], dtype=torch.float32).to(device)
    action_left = torch.tensor(action_left, dtype=torch.float32).to(device)
    action_right = torch.tensor(action_right, dtype=torch.float32).to(device)

    training_data = torch.stack([distance, bearing, omega, pitch, v_x, v_y, action_left, action_right], dim=-1)
    target = wind_along.detach().clone()
    # target = torch.stack([wind_along, wind_vertical], dim=-1)

    '''
    Create the sliding window training data with the targets.
    '''
    windows = []
    for i in range(len(training_data) - WINDOW_SIZE):
        windows.append(training_data[i:i+WINDOW_SIZE])
    training_data = torch.stack(windows).to(device)
    target = target[WINDOW_SIZE:].to(device)
    '''
    Create the data loader
    '''
    dataset = TensorDataset(training_data, target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    epoch_loss = 0
    for batch_x, batch_y in loader:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      optimizer.zero_grad()
      output = model(batch_x)
      loss = loss_function(output, batch_y.unsqueeze(-1))
      loss.backward()
      epoch_loss += loss.item()
      optimizer.step()

    history.append(epoch_loss)

  '''
  Plot out the training loss
  '''
  fig = plt.figure()
  axes = fig.add_subplot(111)
  axes.plot(range(len(history)), history)
  axes.set_title("loss over training epoches")
  plt.show()

  '''
  Saves the model
  '''
  torch.save(model.state_dict(), './vertical_sim/wind_estimator.mdl')

def eval_wind():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE)

  model.load_state_dict(torch.load('./vertical_sim/wind_estimator.mdl', map_location=torch.device("cpu")), strict=False)

  with torch.no_grad():
    testing_data, target = prepare_wind(batch=False) 
    testing_data = torch.detach(testing_data).clone()
    target = torch.detach(target).clone()

    wind_along_preds = []
    wind_along_target = target.detach().clone()

    # wind_vertical_preds = []
    # wind_vertical_target = target[:, 1]
    for data in testing_data:
      data = data.unsqueeze(-1).transpose(0,2).transpose(1,2)                   # Rearranging the data to suit the format of the model inputs (batch, window_size, # features)
      wind_along_preds.append(model(data)[0][0].item())
      # wind_vertical_preds.append(model(data)[0][1].item())
    
    error_along = np.array(wind_along_preds) - np.array(wind_along_target)
    # error_vertical = np.array(wind_vertical_preds) - np.array(wind_vertical_target)
    
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(range(len(wind_along_preds)), wind_along_preds, color="black", alpha=0.3, label="estimated wind")
    axes.plot(range(len(wind_along_target)), wind_along_target, color="red", alpha=0.3, label="true wind")
    axes.set_ylabel("Wind velocity magnitude (m/s)")
    axes.set_xlabel("Time (s)")
    axes.set_title("Wind estimator performance")
    axes.legend()

    fig_err = plt.figure()
    axes_err = fig_err.add_subplot(111)
    axes_err.plot(range(len(error_along)), error_along, color="black", alpha=0.3)
    axes_err.set_ylabel("Error (m/s)")
    axes_err.set_xlabel("Time (s)")
    axes_err.set_title(f"Wind estimator performance - Error over time - Average error is {np.mean(np.array(error_along))}")

    # fig_vertical = plt.figure()
    # axes_vertical = fig_vertical.add_subplot(111)
    # axes_vertical.plot(range(len(wind_vertical_preds)), wind_vertical_preds, color="black", alpha=0.3, label="estimated wind")
    # axes_vertical.plot(range(len(wind_vertical_target)), wind_vertical_target, color="red", alpha=0.3, label="true wind")
    # axes_vertical.set_ylabel("Wind velocity magnitude (m/s)")
    # axes_vertical.set_xlabel("Time (s)")
    # axes_vertical.set_title("Wind estimator performance")
    # axes_vertical.legend()

    # fig_vertical_err = plt.figure()
    # axes_vertical_err = fig_vertical_err.add_subplot(111)
    # axes_vertical_err.plot(range(len(error_vertical)), error_vertical, color="black", alpha=0.3)
    # axes_vertical_err.set_ylabel("Error (m/s)")
    # axes_vertical_err.set_xlabel("Time (s)")
    # axes_vertical_err.set_title(f"Wind estimator performance - Error over time - Average error is {np.mean(np.array(error_along))}")
    plt.show()

def load_wind_estimator():
  model = LSTM_wind_estimator(hidden_dim=HIDDEN_DIM, input_size=INPUT_SIZE).to(device)
  model.load_state_dict(torch.load('./vertical_sims/wind_estimator.mdl', map_location=torch.device("cpu")))
  return model, WINDOW_SIZE

if __name__ == "__main__":
  # train()
  eval_wind()