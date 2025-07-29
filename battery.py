import numpy as np
import math
class Battery():
  '''
  One of the main issues is model mismatch
  '''

  def __init__(self):
    self.capacity = 0
    self.charge_history = []
    self.charge_history.append(self.capacity)

    ## Initiate constants of the drone
    self.draf_coef = 0.011
    self.density = 1.168
    self.rotor_solidity = 0.045 ## defined as h
    self.rotor_radius = 0.26
    self.rotor_area = self.rotor_radius**2*math.pi
    self.k = 0.11 # Incremental correction factor to induced power
    self.weight = 20 # Newtons
    self.thrust_to_weight_ratio = 1
    self.v0 = 6.325 # Mean rotor induced velocity in hover
    self.gamma_h = 0.005256 # Fuselage drag coefficient in horizontal flight 
    self.gamma_v = 0.220168 # Fuselage drag coefficient in vertical flight

    ##########
    # CHANGE #
    ##########

    '''The Thrust coefficient is a function of turbulence as seen in literature review, 
    so this will have to be factored in'''
    self.thrust_coefficient = 0.001195 

  def sgn(self, x):
    if x > 0:
      return 1
    else:
      return -1

  def battery_consumption_h(self, velocity):
    v_planar = np.linalg.norm(np.array([velocity[0], velocity[1]]))
    p_in = (1+self.k)*self.weight**(3/2)/(2*math.sqrt(2*self.density*self.rotor_area))
    return 3/4*self.draf_coef*math.sqrt((self.weight*self.density*self.rotor_area)/self.thrust_coefficient)*self.rotor_solidity*v_planar**2 + p_in*(math.sqrt(math.sqrt(1+v_planar**4/(4*self.v0**4))-v_planar**2/(2*self.v0**2))-1) + 4*self.density*self.gamma_h*v_planar**3

  def battery_consumption_v(self, velocity):
    v_vertical = abs(velocity[2])
    sgn_v = self.sgn(velocity[2])
    return 1/2*self.weight*v_vertical + 2*sgn_v*self.density*self.gamma_v*v_vertical**3 + (self.weight/2+2*sgn_v*self.density*self.gamma_v*v_vertical**2)*(math.sqrt((1+(2*self.gamma_v*sgn_v)/(self.density*self.rotor_area))*v_vertical**2+self.weight/(2*self.density*self.rotor_area)))+self.sgn(v_vertical-1)*self.weight/2*math.sqrt(self.weight/(2*self.density*self.rotor_area))

  def battery_consumption_hover(self):
    return (self.weight/self.thrust_coefficient)**(3/2)*(self.draf_coef*self.rotor_solidity)/(16*math.sqrt(self.density*self.rotor_area))+(1+self.k)*self.weight**(3/2)/(2*math.sqrt(2*self.density*self.rotor_area))

  def battery_consumption(self, velocity, control_frequency):
      energy_loss = 2*self.battery_consumption_hover()/control_frequency + self.battery_consumption_h(velocity=velocity)/control_frequency + self.battery_consumption_v(velocity=velocity)/control_frequency
      self.capacity -= energy_loss
      self.charge_history.append(self.capacity)

    