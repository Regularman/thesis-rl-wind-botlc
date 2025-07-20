from agent_base_class import AgentBase
import numpy as np
import dyanmics as dynamics


class Agent_DSUI(AgentBase):
    def __init__(self,initial_agent_state,initial_agent_estimation):
        super().__init__(initial_agent_state,initial_agent_estimation)

        self.position_estimator_gain = 12.0
        self.velocity_estimator_gain = 6.0

        self.distance_gain = 10

        self.prev_d_correction = 0
        self.discrete_time_gain = 50.0
        # self.dynamics = dynamics.quad_dynamics()


    def act(self,target_state, wind):

        self.calculate_error(target_state)
        self.save_pose_estimation_error(target_state)

        estimated_error = np.linalg.norm(self.position - self.estimated_state[:3]) - self.desired_radius

        # Get bearing measurement 
        bearing = self.calculate_bearing(target_state)
        tangential_bearing = self.perpendicular_vector_clockwise(bearing)

        # Calculate Projection
        Projection_matrix = np.outer(bearing, bearing)
        I = np.eye(3)  # Create a 3x3 identity matrix
        Q = I - Projection_matrix  # Subtract P from the identity matrix

        # Calculate target tuning law
        position_tuning_law = self.position_estimator_gain*Q@(self.position - self.estimated_state[:3]) + self.estimated_state[3:]
        velocity_tuning_law = self.velocity_estimator_gain*Q@(self.position - self.estimated_state[:3])

        # Apply target tuning law
        self.estimated_state[:3] += position_tuning_law / self.control_frequency
        self.estimated_state[3:] += velocity_tuning_law / self.control_frequency
        self.estimated_state_trajectory.append(self.estimated_state)

        d_correction = self.prev_d_correction  + 0.02*self.discrete_time_gain*estimated_error
        output_vector = (self.distance_gain * (estimated_error + d_correction) * bearing  + 
                        (self.tagential_gain - tangential_bearing.transpose() @ self.estimated_state[3:]) * tangential_bearing + 
                        self.estimated_state[3:])
        
        # Apply control action
        self.position = self.position + output_vector / self.control_frequency
        
        self.trajectory.append(self.position)
        self.prev_d_correction = d_correction
        


