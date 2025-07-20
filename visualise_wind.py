from uav_env import UAVEnv 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


env = UAVEnv(render_mode="human")


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection="3d")

env.visualise_wind(ax, fig)

def update(frame):
    ax.clear()  # clear previous plot
    env._update_wind()  # update wind state
    env.visualise_wind(ax=ax, fig=fig)  # re-plot new wind state

# Create the animation
anim = FuncAnimation(fig, update, frames=1000, interval=50)

# # Optional: Save animation to MP4 or GIF
# writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
# anim.save("wind_animation.mp4", writer=writer)

plt.show()
