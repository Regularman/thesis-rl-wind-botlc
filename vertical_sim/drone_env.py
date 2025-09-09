import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from gym_drone_model import *
import os
from event_handler import *
from wind import *
from wind_estimator_final import *

class DroneVertical(gym.Env):
  '''
  Observation is the agent's view of the environment and info is just information used for debugging
  '''
  def __init__(self, render_sim, render_path, render_shade, size, n_steps, desired_distance, frequency, force_scale, p_ground_truth):
  
    self.render_sim = render_sim
    self.render_path = render_path
    self.render_shade = render_shade

    ## Set up the relevant parameters
    self.size = size
    self.maximum_steps = n_steps
    self.desired_distance = desired_distance
    self.force_scale = force_scale
    self.frequency = frequency
    self.drone_shade_distance = 70
    self.p_ground_truth = p_ground_truth

    '''
    Set the uniform distribution range, from 0 to x.
    '''
    self.target_speed = np.random.uniform(0,0.5,size=1).astype(np.float32)
  
    self.wind_strength = 25

    ## Used for real time visualisation
    if self.render_sim is True:
        self.init_pygame()
        self.flight_path = []
        self.drop_path = []
        self.path_drone_shade = []

    self.init_pymunk()
    ## Initial values
    self.info = {}
    self.current_time_step = 0
    self.left_force = -1
    self.right_force = -1
    self.trajectory = []
    self.reward_hist = []
    self.target_trajectory = []
    self.first_step = True
    self.wind_window= [[],[]]
    self.prev_action = [0,0]

    ## Non normalised
    self.distance_hist = []
    self.bearing_hist = []
    self.omega_hist = []
    self.pitch_hist = []
    self.action_left_hist = []
    self.action_right_hist = []
    self.v_x_hist = []
    self.v_y_hist = []
    self.wind_estimations = []

    ## Empirical parameters for velocity normalisation
    self.v_norm = 600
    self.w_norm = 3

    ## Defining spaces for action and observation
    '''
    action[0] = force in the x direction
    action[1] = force in the y direction

    Note that positive thrust constraint has been applied
    '''
    min_action = np.array([-1, -1], dtype=np.float32)
    max_action = np.array([1, 1], dtype=np.float32)
    self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

    self.observation_space = gym.spaces.Dict(
        {
            "v": gym.spaces.Box(-1,1, shape=(2,), dtype=np.float32),           # Linear velocity
            "omega": gym.spaces.Box(-1,1, shape=(1,), dtype=np.float32),       # Angular velocity
            "pitch": gym.spaces.Box(-1,1, shape=(1,), dtype=np.float32),       # Pitch
            "distance": gym.spaces.Box(0,1, shape=(1,), dtype=np.float32),  
            "bearing": gym.spaces.Box(-1,1, shape=(1,), dtype=np.float32),          
            "wind_estimation": gym.spaces.Box(-1,1,shape=(2,), dtype=np.float32)
        }
    )

    self.reset()
  
  def init_pygame(self):
      pygame.init()
      self.screen = pygame.display.set_mode((self.size, self.size))
      pygame.display.set_caption("Drone2d Environment")
      self.clock = pygame.time.Clock()

      script_dir = os.path.dirname(__file__)
      icon_path = os.path.join("img", "icon.png")
      icon_path = os.path.join(script_dir, icon_path)
      pygame.display.set_icon(pygame.image.load(icon_path))

      img_path = os.path.join("img", "shade.png")
      img_path = os.path.join(script_dir, img_path)
      self.shade_image = pygame.image.load(img_path)
  
  def get_obs(self):
    wind_along = 0 

    velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))

    velocity_x = np.clip(velocity_x/self.v_norm, -1, 1)
    velocity_y = np.clip(velocity_y/self.v_norm, -1, 1)
  
    omega = self.drone.frame_shape.body.angular_velocity
    omega = np.clip(omega/self.w_norm, -1, 1)

    alpha = self.drone.frame_shape.body.angle
    alpha = np.clip(alpha/(np.pi/2), -1, 1)
    
    dx = self._target_position[0] - self._agent_position[0]
    dy = self._target_position[1] - self._agent_position[1]
    bearing = math.atan2(dy, dx) / math.pi

    distance = math.sqrt(dx**2+dy**2) /(math.sqrt(2)*self.size)

    counts = np.random.multinomial(1, [self.p_ground_truth, 1-self.p_ground_truth])
    wind_along = self.wind.get_wind(self.current_time_step, self.frequency)[0]
    if counts[0] == 1:  
      wind_along = self.wind.get_wind(self.current_time_step, self.frequency)[0]
    else:
      if len(self.wind_estimations) == 0:
        wind_along = 0
      else: 
        wind_along = self.wind_estimations[-1]

    return {"v": np.array([velocity_x, velocity_y], dtype=np.float32), 
            "omega": np.array([omega], dtype=np.float32),
            "pitch": np.array([alpha], dtype=np.float32), 
            "distance": np.array([distance], dtype=np.float32), 
            "bearing": np.array([bearing], dtype=np.float32),
            "wind_estimation": np.array([wind_along, 0], dtype=np.float32)}    
  
  def get_reward(self, truncated, terminated, obs, action):
    dx = self._target_position[0] - self._agent_position[0]
    dy = self._target_position[1] - self._agent_position[1]

    distance = math.sqrt(dx**2+dy**2)
    reward = (-distance+math.sqrt(2)*self.size)/(math.sqrt(2)*self.size)

    action_left_delta = abs(self.prev_action[0] - (action[0]/2+0.5))
    action_right_delta = abs(self.prev_action[1] - (action[1]/2+0.5))
    if action_left_delta > 0.05:
      reward -= 0.5
    if action_right_delta > 0.05:
       reward -= 0.5

    if distance < self.desired_distance:
      reward += 20*(-distance+self.desired_distance)/self.desired_distance
      if action_left_delta > 0.05:
        reward -= 10
      if action_right_delta > 0.05:
        reward -= 10
    
    # thrust_left = action[0]/2+0.5
    # thrust_right = action[1]/2+0.5

    # reward -= thrust_left**2
    # reward -= thrust_right**2
    
    '''This is negative reward function when the agent flys out of bounds or becomes vertical leading it to drop out of the air'''
    if truncated:
      reward -= 5

    return reward
    
  def step(self, action):
    if self.first_step is True:
        if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
        if self.render_sim is True and self.render_shade is True: self.add_drone_shade()

    # Saving drone's position for drawing
    if self.first_step is True:
        if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
        if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()
        self.first_step = False

    else:
        if self.render_sim is True and self.render_path is True: self.add_postion_to_flight_path()

    if self.render_sim is True and self.render_shade is True:
        x, y = self.drone.frame_shape.body.position
        if np.abs(self.shade_x-x) > self.drone_shade_distance or np.abs(self.shade_y-y) > self.drone_shade_distance:
            self.add_drone_shade()

    terminated = False
    truncated = False
    
    dx = self._target_position[0] - self._agent_position[0]
    dy = self._target_position[1] - self._agent_position[1]
    distance = math.sqrt(dx**2+dy**2)
  
    if self.current_time_step >= self.maximum_steps:
      terminated = True

    ## Updating drone position
    '''
    Remember that action is bounded between 0 and 1. The transformation given moves it to 
    a scale between 0 and 1
    '''
    self.left_force = (action[0]/2+0.5) * self.force_scale
    self.right_force = (action[1]/2+0.5)* self.force_scale
    self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.left_force), (-self.drone_radius, 0))
    self.drone.frame_shape.body.apply_force_at_local_point(Vec2d(0, self.right_force), (self.drone_radius, 0))

    dx = self._target_position[0] - self._agent_position[0]
    dy = self._target_position[1] - self._agent_position[1]
    bearing = math.atan2(dy, dx) / math.pi

    '''
    Updates the wind estimation per step frame
    '''
    try:
      lstm_model, WINDOW_SIZE = load_wind_estimator_optimized()
      if self.current_time_step >= WINDOW_SIZE: 
        data = torch.stack([
            torch.tensor(self.distance_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.bearing_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.omega_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.pitch_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.v_x_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.v_y_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.action_left_hist[-WINDOW_SIZE:], dtype=torch.float32),
            torch.tensor(self.action_right_hist[-WINDOW_SIZE:], dtype=torch.float32)
        ], dim=-1).to(device)

        windows = create_sliding_windows_vectorized(data, WINDOW_SIZE)
        with torch.no_grad():
          prediction = lstm_model(torch.tensor(windows[-1].unsqueeze(-1).transpose(0,2).transpose(1,2))).item()
        '''Low pass filtering'''
        PASS_FILTER_SIZE = 5
        if (len(self.wind_estimations) > PASS_FILTER_SIZE):
            prediction  = (sum(self.wind_estimations[-PASS_FILTER_SIZE:]) + prediction)/(PASS_FILTER_SIZE + 1)
        self.wind_estimations.append(prediction)
    except:
      self.wind_estimations.append(-1)

    ## Distance needs to be normalised by the size of the gymnasium environment hte 
    self.distance_hist.append(distance/self.size)
    self.bearing_hist.append(bearing)
    self.omega_hist.append(self.get_info()["angular_velocity"])
    self.pitch_hist.append(self.get_info()["pitch"])
    self.action_left_hist.append(action[0]/2+0.5)
    self.action_right_hist.append(action[0]/2+0.5)

    self.v_x_hist.append(self.get_obs()["v"][0])
    self.v_y_hist.append(self.get_obs()["v"][1])
    '''
    Updates wind if it is implemented. 
    '''
    wind= self.wind.get_wind(self.current_time_step, self.frequency)
    ##########
    # CHANGE #
    ##########
    ## Wind vertical is currently set to 0
    self.drone.frame_shape.body.velocity += Vec2d(wind[0], 0)           # Boost speed according to the wind

    ## Updates the wind window
    self.wind_window[0].append(wind[0])
    self.wind_window[1].append(wind[1])

    self.space.step(1.0/self.frequency)
    self.current_time_step += 1

    """Update the target position"""
    self._target_position += self._target_velocity
    self.target_trajectory.append(self._target_position)

    self._agent_position = np.array(self.drone.frame_shape.body.position)
    obs = self.get_obs()

    ## If out of bounds or the pitch becomes vertical
    if self._agent_position[0] < 0 or self._agent_position[0] > self.size or self._agent_position[1] < 0 or self._agent_position[1] > self.size:
      truncated = True  
    elif self.get_info()["pitch"] == 1 or self.get_info()["pitch"] == -1:
      truncated = True
    
    reward = self.get_reward(truncated, terminated, obs, action)

    ## Updating the previous force
    self.prev_action[0] = self.left_force
    self.prev_action[1] = self.right_force

    return obs, reward, terminated, truncated, self.get_info()
  
  def render(self, mode="human"):
    if self.render_sim is False: return
    pygame_events(self.space, self, change_target=False)
    self.screen.fill((243, 243, 243))
    pygame.draw.rect(self.screen, (24, 114, 139), pygame.Rect(0, 0, self.size, self.size), 1)

    #Drawing done's shade
    if len(self.path_drone_shade):
        for shade in self.path_drone_shade:
            image_rect_rotated = pygame.transform.rotate(self.shade_image, shade[2]*180.0/np.pi)
            shade_image_rect = image_rect_rotated.get_rect(center=(shade[0], self.size-shade[1]))
            self.screen.blit(image_rect_rotated, shade_image_rect)

    self.space.debug_draw(self.draw_options)

    #Drawing vectors of motor forces
    vector_scale = 0.05
    l_x_1, l_y_1 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, 0))
    l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.force_scale*vector_scale))
    pygame.draw.line(self.screen, (179,179,179), (l_x_1, self.size-l_y_1), (l_x_2, self.size-l_y_2), 1)

    l_x_2, l_y_2 = self.drone.frame_shape.body.local_to_world((-self.drone_radius, self.left_force*vector_scale))
    pygame.draw.line(self.screen, (255,0,0), (l_x_1, self.size-l_y_1), (l_x_2, self.size-l_y_2), 1)

    r_x_1, r_y_1 = self.drone.frame_shape.body.local_to_world((self.drone_radius, 0))
    r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.force_scale*vector_scale))
    pygame.draw.line(self.screen, (179,179,179), (r_x_1, self.size-r_y_1), (r_x_2, self.size-r_y_2), 1)

    r_x_2, r_y_2 = self.drone.frame_shape.body.local_to_world((self.drone_radius, self.right_force*vector_scale))
    pygame.draw.line(self.screen, (255,0,0), (r_x_1, self.size-r_y_1), (r_x_2, self.size-r_y_2), 1)

    pygame.draw.circle(self.screen, (255, 0, 0), (self._target_position[0], self.size-self._target_position[1]), 5)

    #Drawing drone's path
    if len(self.flight_path) > 2:
        pygame.draw.aalines(self.screen, (16, 19, 97), False, self.flight_path)

    if len(self.drop_path) > 2:
        pygame.draw.aalines(self.screen, (255, 0, 0), False, self.drop_path)

    # Return RGB array if in rgb_array mode
    if mode == "rgb_array":
        frame = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
        frame = np.transpose(frame, (1, 0, 2))         # Convert to (H, W, 3)
        return frame
    else:     
      pygame.display.flip()
      self.clock.tick(30)
        
  def get_info(self):
    velocity_x, velocity_y = self.drone.frame_shape.body.velocity_at_local_point((0, 0))
    velocity = math.sqrt(velocity_x**2 + velocity_y**2)
    omega = self.drone.frame_shape.body.angular_velocity

    alpha = self.drone.frame_shape.body.angle
    alpha = np.clip(alpha/(np.pi/2), -1, 1)
    return {"target_position": self._target_position,
            "current_time_step": self.current_time_step,
            "max_timestep": self.maximum_steps,
            "position": np.array([self._agent_position[0], self._agent_position[1]]), 
            "angular_velocity": omega,
            "velocity": velocity, 
            "pitch": alpha,
            "wind": self.wind}

  def reset(self, *, seed=None, options=None):
    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)
    self.trajectory = []
    self.reward_hist = []
    self.target_trajectory = []
    self.current_time_step = 0
    self.wind_window = [[], []]

    wind_strength = np.random.uniform(0,self.wind_strength,size=1).astype(np.float32)
    self.wind = Wind(wind_strength=wind_strength)
  
    self.init_pymunk()
    
    self.wind = Wind(wind_strength=self.wind_strength)
    
    margin = 300
    self._target_position = np.random.uniform(margin, self.size-margin, size=2).astype(np.float32)
  
    # Randomly place target, ensuring it's different from agent position
    while np.array_equal(self._target_position, self._agent_position):
        self._target_position = self.np_random.integers(
            margin, self.size-margin, size=2, dtype=int
        )
  
    """Find the furthest corner and the target will move towards it at various rates"""
    corners = [(0,0), (0,self.size), (self.size, 0), (self.size, self.size)]
    self._target_velocity = np.array([0.0,0.0], dtype=np.float32)
    max_distance = 0
    speed = np.random.uniform(0,self.target_speed,size=1).astype(np.float32)
    for corner in corners:
      distance = (corner[0]-self._target_position[0])**2+(corner[1]-self._target_position[1])**2
      if distance > max_distance:
        max_distance = distance
        ## Calculates the unit vector towards the destination
        self._target_velocity = (corner-self._target_position)/math.sqrt(distance)*speed

    observation = self.get_obs()
    info = self.get_info()
    return observation, info
  
  def add_postion_to_drop_path(self):
      x, y = self.drone.frame_shape.body.position
      self.drop_path.append((x, self.size-y))

  def add_postion_to_flight_path(self):
      x, y = self.drone.frame_shape.body.position
      self.flight_path.append((x, self.size-y))

  def add_drone_shade(self):
      x, y = self.drone.frame_shape.body.position
      self.path_drone_shade.append([x, y, self.drone.frame_shape.body.angle])
      self.shade_x = x
      self.shade_y = y
  
  def close(self):
     pygame.quit()
    
  def init_pymunk(self):
    self.space = pymunk.Space()

    if self.render_sim is True:
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        pymunk.pygame_util.positive_y_is_up = True
    
    self.space.gravity = Vec2d(0, -981)
    #Generating drone's starting position
    random_x = random.uniform(0, self.size)
    random_y = random.uniform(0, self.size)
    angle_rand = random.uniform(0, 0)

    # Randomly place the agent anywhere on the grid
    self._agent_position = np.array([random_x, random_y], dtype=np.float32)
    self.drone = Drone(x=random_x, 
                       y=random_y, 
                       angle=angle_rand, 
                       height=20, 
                       width=100, 
                       mass_f=0.2, 
                       mass_l=0.4, 
                       mass_r=0.4, 
                       space=self.space)

    self.drone_radius = self.drone.drone_radius