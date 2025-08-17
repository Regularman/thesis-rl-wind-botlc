import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from gym_drone_model import *
import os
from event_handler import *
from wind import *

class DroneVertical(gym.Env):
  '''
  Observation is the agent's view of the environment and info is just information used for debugging
  '''
  def __init__(self, render_sim, render_path, render_shade, size, n_steps, desired_distance, frequency, force_scale, wind_strength):
  
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
    '''
    ############
    ## CHANGE ##
    ############
    This is the action for two rotors in the UAV
    '''
    # self.previous_action = [queue.Queue(), queue.Queue()]

    # for i in range(6):
    #   self.previous_action[0].put(0)
    #   self.previous_action[1].put(0)
    self.action_hist = [[], []]
    '''
    ############
    ## CHANGE ##
    ############
    '''
    self.wind_strength = wind_strength
    self.wind = Wind(wind_strength=self.wind_strength)

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
    self.first_step = True

    ## Empirical parameters for velocity normalisation
    self.v_norm = 1330
    self.w_norm = 11.7

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
            "dx": gym.spaces.Box(-1,1, shape=(1,), dtype=np.float32),          # Bearing to the target
            "dy": gym.spaces.Box(-1,1, shape=(1,), dtype=np.float32),
            "position": gym.spaces.Box(-1,1, shape=(2,), dtype=np.float32)
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

    # return np.array([velocity_x, velocity_y, omega, alpha, bearing, self._agent_position[0], self._agent_position[1]])
    x = self._agent_position[0]/self.size
    y = self._agent_position[1]/self.size

    distance = math.sqrt(dx**2+dy**2) / self.size

    return {"v": np.array([velocity_x, velocity_y], dtype=np.float32), 
            "omega": np.array([omega], dtype=np.float32),
            "pitch": np.array([alpha], dtype=np.float32), 
            "dx": np.array([dx/self.size], dtype=np.float32), 
            "dy": np.array([dy/self.size], dtype=np.float32),
            "position": np.array([x,y], dtype=np.float32)}
  
  def get_reward(self, truncated, terminated, obs, action):
    dx = self._target_position[0] - self._agent_position[0]
    dy = self._target_position[1] - self._agent_position[1]
    distance = math.sqrt(dx**2+dy**2)
  
    scale_factor = 5
    b = math.sqrt((5*self.size -1)/scale_factor)
    max_reward = scale_factor*math.sqrt((5*self.size-1)/scale_factor)
    reward = (scale_factor*self.size/(distance+self.size/b))

    '''This is negative reward function when the agent flys out of bounds or becomes vertical leading it to drop out of the air'''
    if truncated:
      reward -= 1

    '''Thrust stabilisation condition'''
    # if terminated or truncated:
    #   max_noise = self.current_time_step
    #   prev_action = 0
    #   action_noise_left = 0
    #   action_noise_right = 0
    #   for action in self.action_hist[0]:
    #     action = action/2+0.5
    #     action_noise_left += abs(action-prev_action)
    #     prev_action = action

    #   prev_action = 0
    #   for action in self.action_hist[1]:
    #     action = action/2+0.5
    #     action_noise_right += abs(action-prev_action)
    #     prev_action = action
      
    #   fluctation_reward = (action_noise_left/max_noise) + (action_noise_right/max_noise)
    #   reward -= fluctation_reward*max_reward

    return reward
    
  def step(self, action):
    if self.first_step is True:
        if self.render_sim is True and self.render_path is True: self.add_postion_to_drop_path()
        if self.render_sim is True and self.render_shade is True: self.add_drone_shade()

    #Saving drone's position for drawing
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

    '''
    Updates wind if it is implemented. 
    '''
    wind= self.wind.get_wind(self.current_time_step, self.frequency)
    self.drone.frame_shape.body.velocity += Vec2d(wind[0], wind[1])           # Boost speed according to the wind

    self.space.step(1.0/self.frequency)
    self.current_time_step += 1


    self._agent_position = np.array(self.drone.frame_shape.body.position)
    obs = self.get_obs()

    if self._agent_position[0] < 0 or self._agent_position[0] > self.size or self._agent_position[1] < 0 or self._agent_position[1] > self.size:
      truncated = True  
    elif obs["pitch"][0] == 1 or obs["pitch"][0] == -1:
      truncated = True
    reward = self.get_reward(truncated, terminated, obs, action)

    ''' 
    Append to the action history 
    '''
    # self.previous_action[0].put(action[0]/2+0.5)
    # self.previous_action[1].put(action[1]/2+0.5)
    # self.previous_action[0].get()
    # self.previous_action[1].get()
    # self.previous_action = np.array([action[0]/2+0.5, action[1]/2+0.5])
    self.action_hist[0].append(action[0]/2+0.5)
    self.action_hist[1].append(action[1]/2+0.5)

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
        frame = np.transpose(frame, (1, 0, 2))        # Convert to (H, W, 3)
        return frame
    else:     
      pygame.display.flip()
      self.clock.tick(60)
        
  def get_info(self):
    return {"target_position": self._target_position,
            "current_time_step": self.current_time_step,
            "max_timestep": self.maximum_steps}

  def reset(self, *, seed=None, options=None):
    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)
    self.trajectory = []
    self.reward_hist = []
    self.current_time_step = 0
    self.action_hist = [[], []]
    self.init_pymunk()
    
    self.wind = Wind(wind_strength=self.wind_strength)
    margin = 1000
    self._target_position = self.np_random.integers(margin, self.size-margin, size=2, dtype=int)

    # Randomly place target, ensuring it's different from agent position
    while np.array_equal(self._target_position, self._agent_position):
        self._target_position = self.np_random.integers(
            margin, self.size-margin, size=2, dtype=int
        )
    observation = self.get_obs()
    info = self.info

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