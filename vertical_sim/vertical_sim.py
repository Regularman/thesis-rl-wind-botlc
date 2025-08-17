from stable_baselines3 import PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from gymnasium.envs.registration import register
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde

register(
    id="drone-2d-custom-v0",
    entry_point="drone_env:DroneVertical",
)

size = 3000
frequency = 60.0
wind_strength = 0
desired_distance = 50
n_steps = 600
total_timesteps = 100000

def step_decay_lr(progress_remaining):
    """
    Step decay based on absolute timestep.
    progress_remaining: 1 at start, 0 at end
    """
    # Convert progress_remaining to current timestep
    current_step = int((1 - progress_remaining) * total_timesteps)
    initial_lr = 0.001
    decay_steps = 50000
    n = current_step // decay_steps
    return initial_lr * (0.5** n)

def train():
  env = Monitor(gym.make('drone-2d-custom-v0', 
                         render_sim = False, 
                         render_path = False, 
                         render_shade = False,
                         size=size, 
                         n_steps=n_steps, 
                         desired_distance=desired_distance,
                         frequency = frequency, 
                         force_scale=1000,
                         wind_strength=wind_strength), filename="./logs/monitor.csv")
  
  model = SAC("MultiInputPolicy", env, verbose=1, action_noise=NormalActionNoise(mean=np.array([0,0]), sigma=np.array([0.1,0.1]), dtype=np.float32), batch_size=128, learning_rate=step_decay_lr)

  ## Note that the frequency is 60
  model.learn(total_timesteps=100000)
  model.save('new_agent')
  
def reward_graph():
  df = pd.read_csv("./logs/monitor.csv", comment="#")
  rewards = df["r"].tolist()

  figure_reward = plt.figure()
  axes_reward = figure_reward.add_subplot(111)
  axes_reward.plot(range(len(rewards)), rewards)
  axes_reward.set_title("Average reward per episode")
  axes_reward.set_xlabel("Episode")
  axes_reward.set_ylabel("Reward")
  plt.show()

def eval(render):
  env = gym.make('drone-2d-custom-v0',
                 render_sim=render,
                 render_path=render,
                 render_shade=False,
                 size=size, 
                 n_steps=600, 
                 desired_distance=desired_distance,
                 frequency = frequency,
                 force_scale=1000,
                 wind_strength=wind_strength)

  model = SAC.load("new_agent.zip", verbose=0)


  model.set_env(env)

  random_seed = int(time.time())
  model.set_random_seed(random_seed)

  obs, info = env.reset()
  trajectory = []
  reward_hist = []
  thrust_left = []
  thrust_right = []
  error = []
  out_of_bounds = False
  failed = False

  try:
    while True:
      
      action, _states = model.predict(obs)
      '''
      Record action to look at thrust demand over the flight time 
      '''
      thrust_left.append(action[0]/2+0.5)
      thrust_right.append(action[1]/2+0.5)
      obs, reward, terminated, truncated, info = env.step(action)
      trajectory.append(obs["position"]*size)
      reward_hist.append(reward)
      error.append(np.linalg.norm(info["target_position"]-obs["position"]*size))
      env.render()
      if terminated or truncated:
        break

  finally:
    if terminated:
      label = "mission successful"
    else:
      label = "mission unsuccessful"
    if render:
      figure = plt.figure()
      axes = figure.add_subplot(111)
      axes.plot(np.array(trajectory)[:,0], np.array(trajectory)[:,1], '-',  alpha=0.5)
      axes.scatter(trajectory[-1][0], trajectory[-1][1], label="END")
      axes.scatter(info["target_position"][0], info["target_position"][1], s=80, marker="X",color="black", label="TARGET")
      axes.set_title(f"{label}")

      figure_thrust = plt.figure()
      axes_thrust = figure_thrust.add_subplot(111)
      axes_thrust.plot(np.linspace(0,int(len(thrust_left)/frequency), len(thrust_left)), thrust_left, alpha=0.3, color="red", label="thrust left rotor")
      axes_thrust.plot(np.linspace(0,int(len(thrust_left)/frequency), len(thrust_left)), thrust_right, alpha=0.3, color="blue", label="thrust right rotor")
      axes_thrust.legend()
      axes_thrust.set_title("Thrust over flight duration")
      axes_thrust.set_xlabel("Time (s)")
      axes_thrust.set_ylabel("Normalised thrust")

      figure_error = plt.figure()
      axes_err = figure_error.add_subplot(111)
      axes_err.plot(range(len(error)), error)
      axes_err.set_title("Distance error of UAV from target over the flight duration")
      plt.show()
  
    if obs["position"][0] < 0 or obs["position"][0] >= size or obs["position"][1] < 0 or obs["position"][1] >= size: 
      out_of_bounds = True
    elif abs(obs["pitch"][0]) ==1:
      failed = True

    env.close()
    return error[-1], failed, out_of_bounds

def calc_average_error():
  iterations = 1000
  error = []
  failed_count = 0
  out_of_bound_count = 0

  for it in tqdm(range(iterations)):
    final_err, failed, out_of_bounds = eval(render=False)
    if failed:
      failed_count += 1
    elif out_of_bounds:
      out_of_bound_count += 1
    else: 
      error.append(final_err)
  average = np.mean(np.array(error))
  print(f"The average steady state error is {average}mm for non truncated case")  
  print(f"The episodes truncates {failed_count+out_of_bound_count} out of {iterations} time. {out_of_bound_count} went out of bounds and {failed_count} went 90 degrees")  

  kde = gaussian_kde(error)
  x_vals = np.linspace(min(error), max(error), 200)  # smooth x-axis
  y_vals = kde(x_vals)

  # Plot KDE
  plt.plot(x_vals, y_vals, color='blue')
  plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
  plt.xlabel("Error")
  plt.ylabel("Density")
  plt.title(f"KDE Curve of Error: Average = {average: .2f}mm over {iterations-failed_count} iterations where it is not truncated. It failed {failed_count+out_of_bound_count} times")
  plt.show()

  # 4. Display the plot
  plt.show()
if __name__ == "__main__":
  # train()
  # reward_graph()
  # calc_average_error()
  eval(render=True)

