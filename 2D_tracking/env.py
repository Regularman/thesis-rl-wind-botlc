import numpy as np

'''Drone Paramaters'''
MASS = 
def drone_dynamics(state, gravity, mass, angular_velocity, thrust):

  state = np.expand_dims(state, axis = 1)

  '''This is the reference z axis used for thrust application'''
  zeta = np.array([0,0,1]).reshape(3,1)
  gravity = np.array([0,0,-1]).reshape(3,1) * gravity
  p = state[0: 3, 0]
  print(p)

drone_dynamics