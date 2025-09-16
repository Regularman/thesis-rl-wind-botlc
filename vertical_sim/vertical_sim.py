from stable_baselines3 import PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from gymnasium.envs.registration import register
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import math
from stable_baselines3.common.callbacks import BaseCallback

from denseMlpPolicy import SACDensePolicy

env_id = "drone-2d-custom-v0"
if env_id not in gym.registry:
  register(
      id=env_id,
      entry_point="drone_env:DroneVertical",
  )

size = 800
frequency = 30.0
desired_distance = 30
n_steps = 1200
total_timesteps = 100000

class EpisodeLimitCallback(BaseCallback):
    """
    Stops training after a specified number of episodes.
    """
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # SB3 automatically stores 'dones' in locals
        dones = self.locals.get('dones')
        if dones is not None:
            self.episode_count += sum(dones)  # Increment for finished episodes
        if self.episode_count >= self.max_episodes:
            if self.verbose > 0:
                print(f"Stopping training at episode {self.episode_count}")
            return False  # Stop training
        return True
  
def step_decay_lr(progress_remaining):
    """
    Step decay based on absolute timestep.
    progress_remaining: 1 at start, 0 at end
    """
    # Convert progress_remaining to current timestep
    current_step = int((1 - progress_remaining) * total_timesteps)
    initial_lr = 0.0003
    decay_steps = 10000
    n = current_step // decay_steps
    return initial_lr * (0.95** n)

def train(p_ground_truth, num_envs):
  def make_env(rank, seed):
      def _init():
          env = gym.make('drone-2d-custom-v0', 
                         render_sim = False, 
                         render_path = False, 
                         render_shade = False,
                         size=size, 
                         n_steps=n_steps, 
                         desired_distance=desired_distance,
                         frequency = frequency, 
                         force_scale=1000,
                         p_ground_truth = p_ground_truth)
          if rank == 0:
            env = Monitor(env, filename=f"./logs/monitor")
          env.reset(seed=seed + rank)
          return env
      return _init
  base_seed = round(time.time())
  env_fns = [make_env(i, base_seed + i) for i in range(num_envs)]
  vec_env = DummyVecEnv(env_fns)
  
  model = SAC(SACDensePolicy, 
              vec_env,
              verbose=1,
              batch_size=256,
              learning_rate=0.0003)
  
  max_episodes = 80000
  episode_callback = EpisodeLimitCallback(max_episodes=max_episodes, verbose=1)
  ## Note that the frequency is 30
  model.learn(total_timesteps=total_timesteps, callback=episode_callback)
  model.save('new_agent')
  
def reward_graph(num_envs):
  rewards = [[] for _ in range(num_envs)]
  for i in range(num_envs):
    df = pd.read_csv(f"./logs/monitor{i}.csv.monitor.csv", comment="#")
    rewards[i] = df["r"].tolist()

  # Find maximum number of episodes
  max_len = max(len(r) for r in rewards)

  # Pad shorter lists with NaN (so we can compute mean safely)
  rewards_padded = [r + [np.nan]*(max_len - len(r)) for r in rewards]

  avg_reward = np.mean(rewards_padded, axis=0)

  figure_reward = plt.figure()
  axes_reward = figure_reward.add_subplot(111)
  axes_reward.plot(range(len(avg_reward)), avg_reward)
  axes_reward.set_title("Average reward per episode")
  axes_reward.set_xlabel("Episode")
  axes_reward.set_ylabel("Reward")
  plt.show()

def eval(render, p_ground_truth, num_envs):
  def make_env(rank, seed=round(time.time())):
      def _init():
          env = gym.make(
              "drone-2d-custom-v0",
              render_sim=True,
              render_path=True,
              render_shade=False,
              size=size,
              n_steps=n_steps,
              desired_distance=desired_distance,
              frequency=frequency,
              force_scale=1000,
              p_ground_truth=p_ground_truth,
          )
          env.reset(seed=seed + rank)  # give each env a different seed
          return env
      return _init
  
  # Create vectorised environment
  env_fns = [make_env(i, seed=round(time.time())) for i in range(num_envs)]
  try:
      vec_env = SubprocVecEnv(env_fns)  # runs envs in parallel subprocesses
  except Exception:
      vec_env = DummyVecEnv(env_fns)    # fallback if env cannot be pickled

  model = SAC.load("./wind_estimator_baseline.zip", 
                   verbose=0, 
                   env=vec_env)
  model.set_env(vec_env)

  vec_env = VecMonitor(vec_env)          # track episode rewards/lengths
  obs = vec_env.reset()

  # one container per env
  trajectory = [[] for _ in range(num_envs)]
  target_trajectory = [[] for _ in range(num_envs)]
  reward_hist = [[] for _ in range(num_envs)]

  thrust_left = [[] for _ in range(num_envs)]
  thrust_right = [[] for _ in range(num_envs)]
  v_x = [[] for _ in range(num_envs)]
  v_y = [[] for _ in range(num_envs)]
  omega = [[] for _ in range(num_envs)]
  bearing = [[] for _ in range(num_envs)]
  wind = [[] for _ in range(num_envs)]
  pitch = [[] for _ in range(num_envs)]
  error = [[] for _ in range(num_envs)]
  along_wind_estimation = [[] for _ in range(num_envs)]
  pitch = [[] for _ in range(num_envs)]

  done_flags = [False] * num_envs  # track which envs are finished

  try:
    while not all(done_flags):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, dones, info = vec_env.step(action)

      for i in range(num_envs):
        if done_flags[i]:
          continue
        if dones[i]:
          done_flags[i] = True
          continue
        '''
        Record action to look at thrust demand over the flight time 
        '''
        thrust_left[i].append(action[i][0]/2+0.5)
        thrust_right[i].append(action[i][1]/2+0.5)

        """
        Load trajectory information to be graphed
        """
        trajectory[i].append(info[i]["position"])
        target_trajectory[i].append(np.array(info[i]["target_position"], copy=True))
        reward_hist[i].append(reward[i])
        omega[i].append(info[i]["angular_velocity"])
        v_x[i].append(obs["v"][i][0])
        v_y[i].append(obs["v"][i][1])
        bearing[i].append(obs["bearing"][i][0])
        error[i].append(np.linalg.norm(info[i]["target_position"]-info[i]["position"]))
        wind[i].append(info[i]["wind"].get_wind(info[i]["current_time_step"], frequency))
        pitch[i].append(obs["pitch"][i][0])
        along_wind_estimation[i].append(obs["wind_estimation"][i][0])

      # Optional: render only the first env
      if render:
        vec_env.env_method("render", indices=[0])

  finally:
    if render:
      fig, axes = plt.subplots(4, 2, figsize=(14, 12))
      axes = axes.flatten()  # flatten for easier indexing

      # 1. Trajectory
      axes[0].plot(np.array(trajectory[0])[:,0], np.array(trajectory[0])[:,1], '-', alpha=0.5)
      axes[0].scatter(trajectory[0][0][0], trajectory[0][0][1], color="green", label="START")
      axes[0].scatter(trajectory[0][-1][0], trajectory[0][-1][1], color="blue", label="END")
      axes[0].plot(np.array(target_trajectory[0])[:, 0], np.array(target_trajectory[0])[:,1], '-', color="red", alpha=0.5)
      axes[0].scatter(info[0]["target_position"][0], info[0]["target_position"][1], 
                      s=20, marker="X", color="black", label="TARGET END")
      axes[0].set_title(f"Trajectory")
      axes[0].legend()
      axes[0].grid(alpha=0.3)

      # 2. Thrust
      time_thrust = np.arange(len(thrust_left[0])) / frequency
      axes[1].plot(time_thrust, thrust_left[0], alpha=0.3, color="red", label="left rotor")
      axes[1].plot(time_thrust, thrust_right[0], alpha=0.3, color="blue", label="right rotor")
      axes[1].set_title("Thrust over flight")
      axes[1].set_xlabel("Time (s)")
      axes[1].set_ylabel("Normalised thrust")
      axes[1].legend()
      axes[1].grid(alpha=0.3)

      # 3. Error
      axes[2].plot(np.arange(len(error[0]))/frequency, error[0])
      axes[2].set_title("Distance error")
      axes[2].set_xlabel("Time (s)")
      axes[2].grid(alpha=0.3)

      # 4. Angular velocity
      axes[3].plot(np.arange(len(omega[0]))/frequency, omega[0])
      axes[3].set_title("Angular velocity")
      axes[3].set_xlabel("Time (s)")
      axes[3].grid(alpha=0.3)

      # 5. Velocity
      vel_mag = np.sqrt(np.array(v_x[0])**2 + np.array(v_y[0])**2)
      axes[4].plot(np.arange(len(v_x[0]))/frequency, vel_mag)
      axes[4].set_title("Velocity magnitude")
      axes[4].set_xlabel("Time (s)")
      axes[4].grid(alpha=0.3)

      # 6. Bearing
      axes[5].plot(np.arange(len(bearing[0]))/frequency, bearing[0])
      axes[5].set_title("Bearing observations")
      axes[5].set_xlabel("Time (s)")
      axes[5].grid(alpha=0.3)

      # 7. Wind
      axes[6].plot(np.arange(len(along_wind_estimation[0]))/frequency, along_wind_estimation[0],
                  color="red", alpha=0.3, label="estimated")
      axes[6].plot(np.arange(len(np.array(wind[0])[:,0]))/frequency, np.array(wind[0])[:,0],
                  color="black", alpha=0.3, label="actual")
      axes[6].set_title("Wind: estimated vs actual")
      axes[6].set_xlabel("Time (s)")
      axes[6].legend()
      axes[6].grid(alpha=0.3)

      # Hide any unused subplot slot (8th panel if 7 plots)
      fig.delaxes(axes[7])

      plt.tight_layout()
      plt.show()

    vec_env.close()
    # Convert lists to np.arrays
    errors = [np.array(e)/size for e in error]
    wind = [np.array(w) for w in wind]
    v_x = [np.array(v) for v in v_x]
    v_y = [np.array(v) for v in v_y]
    omega = [np.array(o) for o in omega]
    pitch = [np.array(p) for p in pitch]
    bearing = [np.array(b) for b in bearing]
    thrust_left = [np.array(t) for t in thrust_left]
    thrust_right = [np.array(t) for t in thrust_right]

    return errors, wind, bearing, v_x, v_y, omega, pitch, thrust_left, thrust_right

if __name__ == "__main__":
  # num_envs = 8
  # train(p_ground_truth=1, num_envs=num_envs)
  # reward_graph(num_envs=num_envs)
  eval(render=True, p_ground_truth=1, num_envs=1)

