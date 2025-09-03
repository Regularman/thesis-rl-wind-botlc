import numpy as np
import dryden_wind
import math

class Wind():
  def __init__(self, wind_strength):
    wind_along, wind_cross, wind_vertical = dryden_wind.dryden_wind_velocities(height=10, airspeed=5)
    ## Wind strength multiplier
    self.wind_along = np.multiply(np.array(wind_along), wind_strength)
    self.wind_cross = np.multiply(np.array(wind_cross), wind_strength)
    self.wind_vertical = np.multiply(np.array(wind_vertical), wind_strength)

  def get_wind(self, step_count, frequency):
    if math.floor(step_count/frequency) < len(self.wind_along) - 1:
        wind_t1 = np.array([self.wind_along[math.floor(step_count/frequency)], self.wind_vertical[math.floor(step_count/frequency)]])
        wind_t2 = np.array([self.wind_along[math.floor(step_count/frequency)+1], self.wind_vertical[math.floor(step_count/frequency)+1]])
        wind = wind_t1 + (step_count%frequency)/frequency*(wind_t2-wind_t1)
    else:
       wind = np.array([0,0])

    '''
    For now, vertical wind is set to 0, and we are only looking at the along wind
    '''
    return np.array([wind[0], 0], dtype=np.float32)
  
  def get_along(self):
     return self.wind_along
  
  def get_vertical(self):
     return self.wind_vertical
  
    
    