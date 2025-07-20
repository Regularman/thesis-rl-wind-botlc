from agent_base_class_2d import AgentBase_2D
import numpy as np



class Agent_DSUI_2D(AgentBase_2D):
    def __init__(self,initial_agent_state,initial_agent_estimation):
        super().__init__(initial_agent_state,initial_agent_estimation)

        self.position_estimator_gain = 12.0
        self.velocity_estimator_gain = 6.0

        self.distance_gain = 10

        self.prev_d_correction = 0
        self.discrete_time_gain = 50.0


    def act(self,target_state):

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
        position_tuning_law = self.position_estimator_gain*Q@(self.position - self.estimated_state[:2]) + self.estimated_state[2:]
        velocity_tuning_law = self.velocity_estimator_gain*Q@(self.position - self.estimated_state[:2])

        # Apply target tuning law
        self.estimated_state[:2] += position_tuning_law / self.control_frequency
        self.estimated_state[2:] += velocity_tuning_law / self.control_frequency
        self.estimated_state_trajectory.append(self.estimated_state)

        d_correction = self.prev_d_correction  + 0.02*self.discrete_time_gain*estimated_error
        output_vector = (self.distance_gain * (estimated_error + d_correction) * bearing  + 
                        (self.tagential_gain - tangential_bearing.transpose() @ self.estimated_state[2:]) * tangential_bearing + 
                        self.estimated_state[2:])
        
        # Apply control action
        self.position = self.position + output_vector / self.control_frequency
        
        self.trajectory.append(self.position)
        self.prev_d_correction = d_correction
        


