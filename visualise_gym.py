from uav_env import UAVEnv 
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import dryden_wind
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def main(wind_strength, desired_radius):
  env = UAVEnv(render_mode="human", desired_radius= desired_radius, wind_strength=wind_strength)
  fig = plt.figure(figsize=(10,10))
  # axes = fig.add_subplot(111, projection="3d")
  axes = fig.add_subplot(111)
  # axes.scatter(env.agent.get_position()[0], env.agent.get_position()[1], env.agent.get_position()[2], label="START")
  axes.scatter(env.agent.get_position()[0], env.agent.get_position()[1], label="START")
  ## This is to create multiple environments for batches processing and increase 
  ## computational efficiency
  # env = DummyVecEnv([lambda: env])

  for it in tqdm(range(int(1000*env.agent.control_frequency))):
    env.step()
  
  trajectory, circum_error_list, wind_estim = env.agent_get_info()
  trajectory = np.array(trajectory)

  # axes.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'x-',  alpha=0.5)
  # axes.scatter(trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], label="END")
  axes.plot(trajectory[:,0], trajectory[:,1], '-',  alpha=0.5)
  axes.scatter(trajectory[-1][0], trajectory[-1][1], label="END")
  axes.scatter(env.target_pos[0], env.target_pos[1], s=80, marker="X",color="black", label="TARGET")
  axes.set_title(f"Wind strength = {wind_strength}, desired radius = {desired_radius}")
  # axes.scatter(env.target_pos[0], env.target_pos[1], env.target_pos[2], s=80, marker="X",color="black", label="TARGET")
  axes.legend()

  circum_error_list = zoom(circum_error_list, 1.0/env.agent.control_frequency)
  
  fig_err = plt.figure()
  axes_err = fig_err.add_subplot(111)
  axes_err.plot(range(len(circum_error_list)), circum_error_list)
  axes_err.set_title("Circumnavigation error")

  fig_wind = plt.figure()
  axes_wind = fig_wind.add_subplot(111)
  along_wind = np.array(wind_estim)[:,0]
  across_wind = np.array(wind_estim)[:,1]

  along_wind = zoom(along_wind, 1.0/env.agent.control_frequency)
  across_wind = zoom(across_wind, 1.0/env.agent.control_frequency)
  axes_wind.plot(range(len(along_wind)), along_wind, color="red", alpha=0.5, label="along wind estimation")
  axes_wind.plot(range(len(across_wind)), across_wind, color="blue", alpha=0.5, label="across wind estimation")
  axes_wind.plot(range(len(env.wind_along)), env.wind_along, color="red", linestyle="--", alpha=1, label="true along wind")
  axes_wind.plot(range(len(env.wind_cross)), env.wind_cross, color="blue", linestyle="--", alpha=1, label="true across wind")
  # axes_wind.plot(range(len(env.wind_along)), np.ones(len(env.wind_along)), color="red", linestyle="--", alpha=1, label="true along wind")
  # axes_wind.plot(range(len(env.wind_cross)), -2*np.ones(len(env.wind_cross)), color="blue", linestyle="--", alpha=1, label="true across wind")
  axes_wind.legend()
  axes_wind.set_title("Estimated vs true wind")
  axes_wind.set_xlabel("Time (s)")
  axes_wind.set_ylabel("wind speed (m/s)")
  
  ## Plot the SOC graph
  axes_battery = env.agent.SOC()

  plt.show()
  

if __name__ == "__main__":
  main(wind_strength=25, desired_radius=2)