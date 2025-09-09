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
from tqdm import tqdm
from scipy.stats import gaussian_kde
import math
from stable_baselines3.common.callbacks import BaseCallback

env_id = "drone-2d-custom-v0"
if env_id not in gym.registry:
  register(
      id=env_id,
      entry_point="drone_env:DroneVertical",
  )

size = 800
frequency = 30.0
desired_distance = 30
n_steps = 1000
total_timesteps = 100000
    
def step_decay_lr(progress_remaining):
    """
    Step decay based on absolute timestep.
    progress_remaining: 1 at start, 0 at end
    """
    # Convert progress_remaining to current timestep
    current_step = int((1 - progress_remaining) * total_timesteps)
    initial_lr = 0.005
    decay_steps = 10000
    n = current_step // decay_steps
    return initial_lr * (0.95** n)

def train():
  env = Monitor(gym.make('drone-2d-custom-v0', 
                         render_sim = False, 
                         render_path = False, 
                         render_shade = False,
                         size=size, 
                         n_steps=n_steps, 
                         desired_distance=desired_distance,
                         frequency = frequency, 
                         force_scale=1000), filename="./logs/monitor.csv")
  
  model = SAC("MultiInputPolicy", 
              env, 
              verbose=1, 
              action_noise=NormalActionNoise(mean=np.array([0,0]), sigma=np.array([0.25,0.25])), 
              learning_rate=step_decay_lr)
  ## Note that the frequency is 30
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

def eval(render, p_ground_truth):
  # env = gym.make('drone-2d-custom-v0',
  #                render_sim=render,
  #                render_path=render,
  #                render_shade=False,
  #                size=size, 
  #                n_steps=n_steps, 
  #                desired_distance=desired_distance,
  #                frequency = frequency,
  #                force_scale=1000,
  #                p_ground_truth=p_ground_truth)
  # model = SAC.load("./new_agent.zip", verbose=0)

  def make_env(render=False, p_ground_truth=1, seed=0):
      def _init():
          env = gym.make(
              'drone-2d-custom-v0',
              render_sim=render,
              render_path=False,
              render_shade=False,
              size=size,
              n_steps=n_steps,
              desired_distance=desired_distance,
              frequency=frequency,
              force_scale=1000,
              p_ground_truth=p_ground_truth
          )
          return env
      return _init
  # Use multiple envs (e.g., 8) for parallel rollout
  n_envs = 8
  env_fns = [make_env(render=False, p_ground_truth=1, seed=time.time()+i) for i in range(n_envs)]
  try:
      vec_env = SubprocVecEnv(env_fns)  # runs envs in parallel subprocesses
  except Exception:
      vec_env = DummyVecEnv(env_fns)    # fallback if env cannot be pickled

  vec_env = VecMonitor(vec_env)          # track episode rewards/lengths


  model = SAC.load("./new_agent.zip", verbose=0, env=vec_env)
  model.set_env(vec_env)

  random_seed = int(time.time())
  model.set_random_seed(random_seed)

  # Track per-env trajectories and other info
  trajectories = [[] for _ in range(n_envs)]
  target_trajectories = [[] for _ in range(n_envs)]
  errors = [[] for _ in range(n_envs)]
  thrust_left = [[] for _ in range(n_envs)]
  thrust_right = [[] for _ in range(n_envs)]

  wind = [[] for _ in range(n_envs)]
  v_x = [[] for _ in range(n_envs)]
  v_y = [[] for _ in range(n_envs)]
  omega = [[] for _ in range(n_envs)]
  pitch = [[] for _ in range(n_envs)]
  bearing = [[] for _ in range(n_envs)]
  along_wind_estimation = [[] for _ in range(n_envs)]

  done_flags = [False for _ in range(n_envs)]
  steps = 0

  obs = vec_env.reset()  # returns a batch of observations  
  while not all(done_flags):
      action, _states = model.predict(obs)
      obs, rewards, dones, infos = vec_env.step(action)
      steps += 1

      for i in range(n_envs):
          if done_flags[i]:
              continue

          # Record data per env
          thrust_left[i].append(action[i][0]/2 + 0.5)
          thrust_right[i].append(action[i][1]/2 + 0.5)
          trajectories[i].append(infos[i]["position"])
          target_trajectories[i].append(np.array(infos[i]["target_position"], copy=True))
          errors[i].append(np.linalg.norm(infos[i]["target_position"] - infos[i]["position"]))
          omega[i].append(infos[i]["angular_velocity"])
          v_x[i].append(obs["v"][i][0])
          v_y[i].append(obs["v"][i][1])
          bearing[i].append(obs["bearing"][i][0])
          wind[i].append(infos[i]["wind"].get_wind(infos[i]["current_time_step"], frequency))
          pitch[i].append(obs["pitch"][i][0])
          along_wind_estimation[i].append(obs["wind_estimation"][i][0])

          if dones[i] or (n_steps and steps >= n_steps):
              done_flags[i] = True

      # Optional: render only the first env
      if render:
        vec_env.env_method("render", indices=[0])

  vec_env.close()

  # Convert lists to np.arrays
  errors = [np.array(e)/size for e in errors]
  wind = [np.array(w) for w in wind]
  v_x = [np.array(v) for v in v_x]
  v_y = [np.array(v) for v in v_y]
  omega = [np.array(o) for o in omega]
  pitch = [np.array(p) for p in pitch]
  bearing = [np.array(b) for b in bearing]
  thrust_left = [np.array(t) for t in thrust_left]
  thrust_right = [np.array(t) for t in thrust_right]

  # obs, info = env.reset()
  # trajectory = []
  # target_trajectory = []
  # reward_hist = []
  # thrust_left = []
  # thrust_right = []
  # error = []
  # v_x = []
  # v_y = []
  # omega = []
  # bearing = []
  # wind = []
  # pitch = []
  # along_wind_estimation = []

  # out_of_bounds = False
  # failed = False

  # try:
  #   while True:
      
  #     # model, WINDOW_SIZE = load_wind_estimator()
  #     # if info["current_time_step"] >= WINDOW_SIZE:
  #     #   pass
  #     action, _states = model.predict(obs)
  #     '''
  #     Record action to look at thrust demand over the flight time 
  #     '''
  #     thrust_left.append(action[0]/2+0.5)
  #     thrust_right.append(action[1]/2+0.5)

  #     obs, reward, terminated, truncated, info = env.step(action)

  #     """
  #     Load trajectory information to be graphed
  #     """
  #     trajectory.append(info["position"])
  #     target_trajectory.append(np.array(info["target_position"], copy=True))
  #     reward_hist.append(reward)
  #     omega.append(info["angular_velocity"])
  #     v_x.append(obs["v"][0])
  #     v_y.append(obs["v"][1])
  #     bearing.append(obs["bearing"][0])
  #     error.append(np.linalg.norm(info["target_position"]-info["position"]))
  #     wind.append(info["wind"].get_wind(info["current_time_step"], frequency))
  #     pitch.append(obs["pitch"][0])
  #     along_wind_estimation.append(obs["wind_estimation"][0])
  #     env.render()

  #     if terminated or truncated:
  #       break

  # finally:
  #   if terminated:
  #     label = "mission successful"
  #   else:
  #     label = "mission unsuccessful"
  if render:
    i = 0  # first environment
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    traj = np.array(trajectories[i])
    target = np.array(target_trajectories[i])
    env_error = np.array(errors[i])
    env_thrust_left = np.array(thrust_left[i])
    env_thrust_right = np.array(thrust_right[i])
    env_omega = np.array(omega[i])
    env_vel_x = np.array(v_x[i])
    env_vel_y = np.array(v_y[i])
    env_bearing = np.array(bearing[i])
    env_wind = np.array(wind[i])
    env_pitch = np.array(pitch[i])
    env_along_wind = np.array(along_wind_estimation[i])
    
    label = "Env 1"

    # 1. Trajectory
    axes[0].plot(traj[:,0], traj[:,1], '-', alpha=0.5)
    axes[0].scatter(traj[0,0], traj[0,1], color="green", label="START")
    axes[0].scatter(traj[-1,0], traj[-1,1], color="blue", label="END")
    axes[0].plot(target[:,0], target[:,1], '-', color="red", alpha=0.5)
    axes[0].set_title(f"Trajectory ({label})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Thrust
    time_thrust = np.arange(len(env_thrust_left)) / frequency
    axes[1].plot(time_thrust, env_thrust_left, alpha=0.3, color="red", label="left rotor")
    axes[1].plot(time_thrust, env_thrust_right, alpha=0.3, color="blue", label="right rotor")
    axes[1].set_title("Thrust over flight")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalised thrust")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Distance error
    axes[2].plot(np.arange(len(env_error))/frequency, env_error)
    axes[2].set_title("Distance error")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    # 4. Angular velocity
    axes[3].plot(np.arange(len(env_omega))/frequency, env_omega)
    axes[3].set_title("Angular velocity")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.3)

    # 5. Velocity magnitude
    vel_mag = np.sqrt(env_vel_x**2 + env_vel_y**2)
    axes[4].plot(np.arange(len(env_vel_x))/frequency, vel_mag)
    axes[4].set_title("Velocity magnitude")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(alpha=0.3)

    # 6. Bearing
    axes[5].plot(np.arange(len(env_bearing))/frequency, env_bearing)
    axes[5].set_title("Bearing observations")
    axes[5].set_xlabel("Time (s)")
    axes[5].grid(alpha=0.3)

    # 7. Wind
    axes[6].plot(np.arange(len(env_along_wind))/frequency, env_along_wind,
                 color="red", alpha=0.3, label="estimated")
    axes[6].plot(np.arange(len(env_wind))/frequency, env_wind,
                 color="black", alpha=0.3, label="actual")
    axes[6].set_title("Wind: estimated vs actual")
    axes[6].set_xlabel("Time (s)")
    axes[6].legend()
    axes[6].grid(alpha=0.3)

    # 8th subplot: Pitch
    axes[7].plot(np.arange(len(env_pitch))/frequency, env_pitch)
    axes[7].set_title("Pitch")
    axes[7].set_xlabel("Time (s)")
    axes[7].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

  return {
      "errors": errors,
      "trajectories": trajectories,
      "target_trajectories": target_trajectories,
      "thrust_left": thrust_left,
      "thrust_right": thrust_right,
      "wind": wind,
      "v_x": v_x,
      "v_y": v_y,
      "omega": omega,
      "pitch": pitch,
      "bearing": bearing
  }
  #     # Hide any unused subplot slot (8th panel if 7 plots)
  #     fig.delaxes(axes[7])

  #     plt.tight_layout()
  #     plt.show()

  #   if info["position"][0] < 0 or info["position"][0] >= size or info["position"][1] < 0 or info["position"][1] >= size: 
  #     out_of_bounds = True
  #   elif abs(obs["pitch"][0]) ==1:
  #     failed = True

  #   env.close()
  #   return np.array(error)/size, failed, out_of_bounds, np.array(wind), np.array(bearing), np.array(v_x), np.array(v_y), np.array(omega), np.array(pitch), np.array(thrust_left), np.array(thrust_right)

def calc_average_error():
  iterations = 1000
  error = []
  failed_count = 0
  out_of_bound_count = 0

  for it in tqdm(range(iterations)):
    final_err, failed, out_of_bounds, wind, bearing, velocity, omega, pitch= eval(render=False)
    if failed:
      failed_count += 1
    elif out_of_bounds:
      out_of_bound_count += 1
    else: 
      error.append(final_err[-1])
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
  eval(render=True, p_ground_truth=1)

