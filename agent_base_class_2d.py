import numpy as np
import math


class AgentBase_2D:
    def __init__(self,initial_agent_state=None,initial_agent_estimation=None):

        if initial_agent_state is not None:
            self.position = np.array(initial_agent_state[:2])
            self.velocity = np.array(initial_agent_state[2:])
        else:
            self.position = np.array([np.random.uniform(-10, 15), np.random.uniform(-15, 15)])
            self.velocity = np.array([0,0])

        if initial_agent_estimation is not None:
            self.estimated_state = np.array(initial_agent_estimation)
        else:
            self.estimated_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.estimated_state_trajectory = []
        self.estimated_state_trajectory.append(self.estimated_state)



        self.desired_radius = 10
        self.control_frequency = 50

        self.tagential_gain = 60
        self.distance_gain = 10

        self.trajectory = []
        self.trajectory.append(self.position)
        self.distance_error_list = []
        self.pose_estimation_error_list = []
        self.circumnavigation_error_list = []



    def act(self, *args, **kwargs):
        raise NotImplementedError("The act method must be implemented in the derived class.")
    
    def reset(self,initial_agent_state,initial_agent_estimation):
        self.position = np.array(initial_agent_state[:2])
        self.velocity = np.array(initial_agent_state[2:])

        self.estimated_state = np.array(initial_agent_estimation)
        self.estimated_state_trajectory = []
        self.estimated_state_trajectory.append(self.estimated_state)

        self.trajectory = []
        self.distance_error_list = []
        self.pose_estimation_error_list = []
        self.circumnavigation_error_list = []
    def get_state(self):
        return [self.position, self.velocity, self.yaw]
    
    def get_position(self):
        return self.position
    
    def set_radius(self, radius):
        self.desired_radius = radius
    
    def perpendicular_vector_clockwise(self, bearing):
        perpendicular_x = bearing[1]
        perpendicular_y = -bearing[0]
        
        return np.array([perpendicular_x, perpendicular_y])
    
    def calculate_bearing(self, target_state):
        target_x, target_y = target_state[:2]
        agent_x, agent_y = self.position
        dx = target_x - agent_x
        dy = target_y - agent_y
        bearing = math.atan2(dy, dx)


        vector_x = math.cos(bearing)
        vector_y = math.sin(bearing)

        return np.array([vector_x,vector_y])
    
    def save_pose_estimation_error(self,target_state):
        pose_estimation_error = np.linalg.norm(target_state[:2] - self.estimated_state[:2])
        self.pose_estimation_error_list.append(pose_estimation_error)
    
    def get_estimated_state_error(self,target_state):
        estimated_target_state_error = target_state - self.estimated_state
        return estimated_target_state_error
    
    def calculate_error(self, target_state):
        agent_to_target_vector = np.array([target_state[0] - self.position[0], target_state[1] - self.position[1]])
        current_distance = np.linalg.norm(agent_to_target_vector)
        radius_error = current_distance

        self.distance_error_list.append(radius_error)
        self.circumnavigation_error_list.append(abs(self.desired_radius - radius_error))

    def check_if_agent_has_lost_tracking(self,step):

        if step < 100: # Wait for the agent to stabilize
            return False

        if len(self.distance_error_list) >= 30:
            last_ten_errors = self.distance_error_list[-30:]
            if all(error > 40 for error in last_ten_errors):
                return True 


        
        if len(self.distance_error_list) >= 50:
            last_ten_errors = self.distance_error_list[-50:]
            if all(error < 5 for error in last_ten_errors):
                return True 
        return False  
