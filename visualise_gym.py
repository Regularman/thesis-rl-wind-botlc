from uav_env import UAVEnv 
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import dryden_wind
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def main():
  env = UAVEnv(render_mode="human")
  fig = plt.figure(figsize=(10,10))
  # axes = fig.add_subplot(111, projection="3d")
  axes = fig.add_subplot(111)
  # axes.scatter(env.agent.get_position()[0], env.agent.get_position()[1], env.agent.get_position()[2], label="START")
  axes.scatter(env.agent.get_position()[0], env.agent.get_position()[1], label="START")
  ## This is to create multiple environments for batches processing and increase 
  ## computational efficiency
  # env = DummyVecEnv([lambda: env])

  for it in tqdm(range(20)):
    env.step()
  
  trajectory, distance_err_list = env.agent_get_info()
  trajectory = np.array(trajectory)

  # axes.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'x-',  alpha=0.5)
  # axes.scatter(trajectory[-1][0], trajectory[-1][1], trajectory[-1][2], label="END")
  axes.plot(trajectory[:,0], trajectory[:,1], 'x-',  alpha=0.5)
  axes.scatter(trajectory[-1][0], trajectory[-1][1], label="END")
  axes.scatter(env.target_pos[0], env.target_pos[1], s=80, marker="X",color="black", label="TARGET")
  # axes.scatter(env.target_pos[0], env.target_pos[1], env.target_pos[2], s=80, marker="X",color="black", label="TARGET")
  axes.legend()

  fig_err = plt.figure()
  axes_err = fig_err.add_subplot(111)
  axes_err.plot(range(len(distance_err_list)), distance_err_list)
  plt.show()

if __name__ == "__main__":
  main()