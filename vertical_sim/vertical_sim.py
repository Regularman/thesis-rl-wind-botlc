from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium.envs.registration import register
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

register(
    id="drone-2d-custom-v0",
    entry_point="drone_env:DroneVertical",
)

size = 800
frequency = 60.0

def train():
  env = Monitor(gym.make('drone-2d-custom-v0', 
                         render_sim = False, 
                         render_path = False, 
                         render_shade = False,
                         size=size, 
                         n_steps=500, 
                         desired_distance=50,
                         frequency = frequency, 
                         force_scale=1000), filename="./logs/monitor.csv")

  model = SAC("MultiInputPolicy", env, verbose=1)

  ## Note that the frequency is 60
  model.learn(total_timesteps=180000)
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

def eval():
  env = gym.make('drone-2d-custom-v0',
                 render_sim=True,
                 render_path=True,
                 render_shade=False,
                 size=size, 
                 n_steps=500, 
                 desired_distance=50,
                 frequency = frequency,
                 force_scale=1000)

  """
  The example agent used here was originally trained with Python 3.7
  For this reason, it is not compatible with Python version >= 3.8
  Agent has been adapted to run in the newer version of Python,
  but because of this, you cannot easily resume their training.
  If you are interested in resuming learning, please use Python 3.7.
  """

  model = SAC.load("new_agent.zip")


  model.set_env(env)

  random_seed = int(time.time())
  model.set_random_seed(random_seed)

  obs, info = env.reset()
  trajectory = []
  reward_hist = []
  thrust_left = []
  thrust_right = []
  error = []
  try:
    while True:
      
      action, _states = model.predict(obs)
      '''
      Record action to look at thrust demand over the flight time 
      '''
      thrust_left.append(action[0])
      thrust_right.append(action[1])

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
    axes
    plt.show()
    env.close()

if __name__ == "__main__":
  # train()
  eval()
