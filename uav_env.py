import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, List, Any, Optional
import numpy.typing as npt
from dataclasses import dataclass
import dryden_wind
from agent_dsui import Agent_DSUI
from agent_dsui_2d_wind_estimator import Agent_DSUI_wind_2D
from agent_dsui_2d import Agent_DSUI_2D
import math
import time
import warnings

class UAVEnv(gym.Env):
    """A custom Gymnasium environment for UAV navigation in 3D space with turbulent wind."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, wind_strength, desired_radius, render_mode: Optional[str] = None) -> None:
        """Initialize the UAV environment.
        
        Args:
            render_mode (Optional[str]): The render mode to use. Defaults to None.
        """
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -2, -2, -2], dtype=np.float32),
            high=np.array([20, 20, 20, 2, 2, 2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Environment parameters
        self.grid_size = 20
        self.start_pos = np.array([0., 0., 0.], dtype=np.float32)
        self.target_pos = np.array([18., 18., 18.], dtype=np.float32)
        self.max_steps = 200  # Maximum episode length
        self.target_threshold = 1.0  # Distance threshold for reaching target
        
        # Energy parameters
        ############
        ## CHANGE ##
        ############
        self.base_energy_cost = 0.1  # Base energy cost for any movement
        self.wind_resistance_factor = 0.5  # Factor for wind resistance energy cost
        self.energy_penalty_weight = 0.2  # Weight for energy consumption in reward

        # Initialize state
        self.current_pos: Optional[npt.NDArray[np.float32]] = None
        self.step_count: int = 0
        self.total_energy_consumed: float = 0.0
        
        # Rendering setup
        self.render_mode = render_mode
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        
        # Add in agents
        self.agent = Agent_DSUI_wind_2D(desired_radius=desired_radius, 
                                   initial_agent_state=np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]), 
                                   initial_agent_estimation=[17.0, 17.0, 0.0, 0.0])

        # Add in target (Assume stationary target for now)
        ############
        ## CHANGE ##
        ############

        # Initialize the environment
        self.reset(wind_multiplier=wind_strength)
    
    def reset(self, *, wind_multiplier, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed (Optional[int]): The seed for random number generation.
            options (Optional[Dict]): Additional options for reset.
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        self.current_pos = self.start_pos.copy()
        self.step_count = 0
        self.total_energy_consumed = 0.0

        # Initialize wind field with more realistic turbulent components
        self.wind = np.zeros((self.grid_size, self.grid_size, self.grid_size, 3), dtype=np.float32)
        wind_along, wind_cross, wind_vertical = dryden_wind.dryden_wind_velocities(height=10, airspeed=5)
        self.wind_along = np.multiply(np.array(wind_along), wind_multiplier)
        self.wind_cross = np.multiply(np.array(wind_cross), wind_multiplier)
        self.wind_vertical = np.multiply(np.array(wind_vertical), wind_multiplier)
        
        max_wind = 0

        for i, wind in enumerate(self.wind_along):
            mag = math.sqrt(self.wind_along[i]**2 + self.wind_cross[i]**2 + self.wind_vertical[i]**2)
            if mag> max_wind:
                max_wind = mag
        self.max_wind = max_wind
    
        self._update_wind()
        
        observation = self._get_obs()
        info = self._get_info()
        
        # if self.render_mode == "human":
        #     self._render_frame()
            
        return observation, info
    
    def step(self):
        """Execute one time step within the environment.
        
        Args:
            action (np.ndarray): The action to take.
        """
        frequency = self.agent.control_frequency

        ## Treat wind as linear interpolation between timesteps
        if math.floor(self.step_count/frequency) < len(self.wind_along) - 1:
            wind_t1 = self.wind_along[math.floor(self.step_count/frequency)]
            wind_t2 = self.wind_along[math.floor(self.step_count/frequency) + 1]
            wind = wind_t1 + self.step_count%frequency*(wind_t2-wind_t1)
        else:
            wind = np.array([0,0])
        self.agent.act(np.array([self.target_pos[0], self.target_pos[1]]), wind)
        if frequency % 1000 == 0: 
            self._update_wind()
        self.step_count += 1
        

    def agent_get_info(self):
        return self.agent.trajectory, self.agent.circumnavigation_error_list, self.agent.wind_estimation

    def _get_obs(self) -> npt.NDArray[np.float32]:
        """Get current observation.
        
        Returns:
            np.ndarray: The current observation (position and wind).
        """
        wind = self._get_wind_at_pos(self.current_pos)
        return np.concatenate([self.current_pos, wind]).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info.
        
        Returns:
            Dict[str, Any]: Information about the current state.
        """
        return {
            "distance_to_target": np.linalg.norm(self.current_pos - self.target_pos),
            "step_count": self.step_count,
            "total_energy_consumed": self.total_energy_consumed,
            "current_wind": self._get_wind_at_pos(self.current_pos)
        }
    
    def _get_wind_at_pos(self, pos):
        """Get wind vector at given position.
        
        Args:
            pos (np.ndarray): Position to get wind at.
            
        Returns:
            np.ndarray: Wind vector at the given position.
        """
        x, y, z = np.clip(pos.astype(int), 0, self.grid_size - 1)
        return self.wind[x, y, z]
    
    def _is_within_bounds(self, pos: npt.NDArray[np.float32]) -> bool:
        """Check if position is within grid bounds.
        
        Args:
            pos (np.ndarray): Position to check.
            
        Returns:
            bool: True if position is within bounds.
        """
        return np.all(pos >= 0) and np.all(pos < self.grid_size)
    
    def _update_wind(self) -> None:
        """ Updating according to the Dryden turbulence model stored in the environment variable, 
        if turbulence time frame is exceeded, then wind will remain static
        """
        if self.step_count < len(self.wind_along):
            for i in range(0, self.grid_size):
                for j in range(0, self.grid_size):
                    for k in range(0, self.grid_size):
                        self.wind[i,j,k] = np.array([self.wind_along[self.step_count], 
                                            self.wind_cross[self.step_count], 
                                            self.wind_vertical[self.step_count]])
    
    def _render_frame(self) -> None:
        """Render the current frame."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Plot UAV position
        self.ax.scatter(self.current_pos[0], self.current_pos[1], self.current_pos[2], 
                       c='blue', marker='o', s=100, label='UAV')
        
        # Plot target position
        self.ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], 
                       c='green', marker='*', s=200, label='Target')
        
        # Plot wind vectors (subsampled)
        step = 4  # Plot every 4th point
        for i in range(0, self.grid_size, step):
            for j in range(0, self.grid_size, step):
                for k in range(0, self.grid_size, step):
                    wind = self.wind[i, j, k]
                    if np.linalg.norm(wind) > 0.1:  # Only plot significant wind
                        self.ax.quiver(i, j, k, 
                                     wind[0], wind[1], wind[2],
                                     color='gray', alpha=0.3,
                                     length=0.5, normalize=True)
        
        # Set plot limits and labels
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_zlim(0, self.grid_size)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # Update the display
        plt.draw()
        plt.pause(0.01)
    
    def visualise_wind(self, ax,fig,): 

        if ax is None:
            fig = plt.figure(figsize=(10,10))
            axes = fig.add_subplot(111, projection="3d")
        else: 
            axes = ax

        positions = []
        directions = []
        magnitudes = []
        # Plot wind vectors (subsampled)
        step = 4  # Plot every 4th point             

        for i in range(0, self.grid_size, step):
            for j in range(0, self.grid_size, step):
                for k in range(0, self.grid_size, step):
                    wind = self.wind[i,j,k]
                    positions.append((i, j, k))
                    magnitude = np.linalg.norm(wind) 
                    directions.append(np.array([wind[0]/magnitude, wind[1]/magnitude, wind[2]/magnitude]))
                    magnitudes.append(magnitude)
        positions = np.array(positions)
        directions = np.array(directions)
        magnitudes = np.array(magnitudes)

        # Get colormap colors based on normalized magnitude on an abolsute scale
        vmin = 0
        vmax = math.ceil(self.max_wind) # You can adjust based on expected wind max
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.inferno
        colors_mapped = cmap(norm(magnitudes))

        axes.quiver(
            positions[:,0], positions[:,1], positions[:,2], 
            directions[:,0], directions[:,1], directions[:,2],
            length=1.5,  
            color=colors_mapped,
            alpha=1, 
            linewidth=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar
        # Only add colorbar if it doesn't already exist
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            self.colorbar = fig.colorbar(sm, ax=ax, label='Wind Magnitude (m/s)', shrink=0.25)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            Optional[np.ndarray]: RGB array of the rendered frame if mode is "rgb_array".
        """
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            if self.fig is None:
                self._render_frame()
            return np.array(self.fig.canvas.renderer.buffer_rgba())
        return None  # Add rgb_array support if needed
    
    def close(self) -> None:
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
