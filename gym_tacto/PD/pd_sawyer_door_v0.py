import numpy as np

class PDController():
    def __init__(self, env):
        self.env = env
        self.reset()

    def set_default_values(self):
        self.error = 0.0005
        self.k_p = 15
        self.k_d = 0.05
        self.dt = 1./240
        self.state = 0

    def reset(self):
        self.set_default_values()
        self.init_target()

    def init_target(self):
        target_position = self.env.get_door_handle_position()
        gripper_width = self.env.get_gripper_width()
        self.target = [target_position[0] - 0.125, target_position[1], target_position[2] - 0.125, gripper_width]    

    def get_relative_observation(self):
        observation = self.env.get_current_state()
        dx = self.target[0] - observation["position"][0]
        dy = self.target[1] - observation["position"][1]
        dz = self.target[2] - observation["position"][2]
        dgw = self.target[3] - observation["gripper_width"]
        return (dx, dy, dz, dgw)

    def change_target(self, dx, dy, dz, dgw):
        if abs(dx) < self.error and abs(dy) < self.error and  abs(dz) < self.error and abs(dgw) < self.error:
            self.state += 1
            target_position = self.env.get_door_handle_position()
            gripper_width = self.env.get_gripper_width()
            if self.state == 1:
                self.target = [target_position[0] - 0.125, target_position[1], target_position[2] - 0.125, 0.045]   
            elif self.state == 2:
                self.target = [target_position[0] - 0.125, target_position[1], target_position[2] - 0.175, 0.045]   
    def clamp_action(self, action):
        # Assure every action component is scaled between -1, 1
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action 
        return action

    def get_action(self):   
        dx, dy, dz, dgw = self.get_relative_observation()
        
        pd_x = self.k_p*dx + self.k_d*dx/self.dt
        pd_y = self.k_p*dy + self.k_d*dy/self.dt
        pd_z = self.k_p*dz + self.k_d*dz/self.dt
        pd_gw = self.k_p*dgw + self.k_d*dgw/self.dt

        self.change_target(dx, dy, dz, dgw)
        action = np.array([pd_x, pd_y, pd_z, pd_gw], dtype=float)

        return self.clamp_action(action) 