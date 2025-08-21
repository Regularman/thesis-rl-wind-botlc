from agent_base_class_2d import AgentBase_2D
import numpy as np
import matplotlib.pyplot as plt

class Agent_DSUI_wind_2D(AgentBase_2D):
    def __init__(self,desired_radius, initial_agent_state,initial_agent_estimation):
        super().__init__(desired_radius, initial_agent_state,initial_agent_estimation)

        self.position_estimator_gain = 3.2
        self.wind_estimator_gain = 5
        self.control_gain_c = 5
        self.control_gain_eta = 3
        self.max_velocity = 5 ## m/s

        # self.distance_gain = 10
        self.velocity_estimator_gain = 2.5
        # self.prev_d_correction = 0
        # self.discrete_time_gain = 50.0

    def wind_estimator(self, bearing):
      estimated_wind = -self.wind_estimator_gain*(np.linalg.norm(self.estimated_state[:2] - self.position) - self.desired_radius)*bearing
      return estimated_wind

    def act(self,target_state, wind):

        self.calculate_error(target_state)
        self.save_pose_estimation_error(target_state)

        estimated_error = np.linalg.norm(self.position - self.estimated_state[:2]) - self.desired_radius

        # Get bearing measurement 
        bearing = self.calculate_bearing(target_state)
        tangential_bearing = self.perpendicular_vector_clockwise(bearing)

        # Calculate Projection
        Projection_matrix = np.outer(bearing, bearing)
        I = np.eye(2)  # Create a 2x2 identity matrix
        Q = I - Projection_matrix  # Subtract P from the identity matrix

        # Calculate target tuning law
        position_tuning_law = self.position_estimator_gain*Q@(self.position - self.estimated_state[:2])
        velocity_tuning_law = self.velocity_estimator_gain*Q@(self.position - self.estimated_state[:2])
        wind_tuning_law = self.wind_estimator(bearing)


        # Apply target tuning law
        self.estimated_state[:2] += position_tuning_law / self.control_frequency
        self.estimated_state[2:] += velocity_tuning_law / self.control_frequency
        self.wind_estimation = self.wind_estimation + wind_tuning_law / self.control_frequency
        
        self.estimated_state_trajectory.append(self.estimated_state)
        self.wind_estimation_hist.append(self.wind_estimation)

        output_vector = self.control_gain_c * estimated_error * bearing + (self.control_gain_eta + tangential_bearing.transpose()@self.wind_estimation)*tangential_bearing - self.wind_estimation

        ## Impose speed constraint
        if np.linalg.norm(output_vector) > self.max_velocity:
          output_vector = output_vector / np.linalg.norm(output_vector) * self.max_velocity
        # Apply control action
        self.position = self.position + output_vector / self.control_frequency  + wind / self.control_frequency
        # Update battery state of charge (You need the relative velocity against the wind)
        self.battery_model.battery_consumption(velocity=np.array([output_vector[0], output_vector[1], 0]), control_frequency=self.control_frequency)       
        
        self.trajectory.append(self.position)

            