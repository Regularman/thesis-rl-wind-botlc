from uav_env import UAVEnv 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import math

def animate():
    env = UAVEnv(desired_radius=10, wind_strength=10, render_mode="human")


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

def wind_plots(wind_strength):
    env = UAVEnv(desired_radius=10, wind_strength=wind_strength, render_mode="human")
    wind_along = env.wind_along
    wind_cross = env.wind_cross
    wind_vertical = env.wind_vertical
    magnitude = []
    for i in range(len(wind_along)):
        magnitude.append(math.sqrt(wind_along[i]**2+wind_cross[i]**2+wind_vertical[i]**2))

    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot()
    axes.plot(range(len(magnitude)), magnitude, color="black", label="wind magnitude (m/s)")
    axes.plot(range(len(magnitude)), wind_along, color="red", alpha=0.7, label="along wind magnitude (m/s)")
    axes.plot(range(len(magnitude)), wind_cross, color="blue", alpha=0.7, label="cross wind magnitude (m/s)")
    axes.plot(range(len(magnitude)), wind_vertical, color="green", alpha=0.7, label="vertical wind magnitude (m/s)")
    axes.set_title("Wing magnitude over time")
    axes.set_xlabel("Time-step (s)")
    axes.set_ylabel("Magnitude (m/s)")
    axes.legend()
    plt.show()
if __name__ == "__main__":
    wind_plots(wind_strength=35)
