import hydra
import gym
from gym import spaces, ObservationWrapper, ActionWrapper
import pybullet as p
import numpy as np
import logging
import tacto
import pybulletX as px
from pybulletX.utils.space_dict import SpaceDict
from gym_tacto.envs.sawyer_gripper import SawyerGripper
from gym_tacto.envs.cameras.static_camera import StaticCamera
from gym_tacto.utils.path import pkg_path

class SawyerDoorEnv(gym.Env):
    metadata = {'render.modes': ['view_1', 'view_2', 'all_views']}

    def __init__(self, cfg):
        """ 
        Input: cfg contains the custom configuration of the environment
        cfg.tacto 
        cfg.settings
            show_gui=False, 
            dt=0.005, 
            action_frequency=30, 
            simulation_frequency=240, 
            max_episode_steps=500,
        """
        # Init logger
        self.logger = logging.getLogger(__name__)

        # Init logic parameters
        self.cfg = cfg
        self.show_gui = cfg.settings.show_gui
        self.dt = cfg.settings.dt
        self.action_frequency = cfg.settings.action_frequency
        self.simulation_frequency = cfg.settings.simulation_frequency
        self.max_episode_steps = cfg.settings.max_episode_steps
        self.random_obj_position = cfg.settings.random_obj_position

        # Set interaction parameters
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        # Init environment
        self.logger.info("Initializing world")
        mode = p.GUI if self.show_gui else p.DIRECT
        client_id = px.init(mode=mode) 
        self.physics_client = px.Client(client_id=client_id)
        p.resetDebugVisualizerCamera(**cfg.debug_camera)
        p.setTimeStep(1/self.simulation_frequency)
        self.init_cameras()

        p.resetSimulation()
        if self.show_gui:
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Disable explorer menu
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Disable rendering during setup
        self.load_objects()
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0,0,-9.8)

    @staticmethod
    def get_observation_space():
        """Return only position and gripper_width by default"""
        observation_space = {}
        observation_space["position"] = spaces.Box(
                low=np.array([0.3, -0.85, 0]), high=np.array([0.85, 0.85, 0.8]))
        observation_space["gripper_width"] = spaces.Box(low=0.03, high=0.11, shape=(1,))
        return SpaceDict(observation_space)

    @staticmethod
    def get_action_space():
        """End effector position and gripper width relative displacement"""      
        return spaces.Box(np.array([-1]*4), np.array([1]*4))

    @staticmethod
    def clamp_action(action):
        """Assure every action component is scaled between -1, 1"""
        max_action = np.max(np.abs(action))
        if max_action > 1:
            action /= max_action
        return action

    def reset_initial_positions(self):
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.robot.reset()
        self.door.reset()
        p.setJointMotorControl2(self.door.id, 1, controlMode=p.POSITION_CONTROL, 
                                targetPosition=0, force=10)
        self.lock_door()
        if self.show_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def reset(self):
        self.reset_initial_positions()
        state = self.get_current_state()
        return state

    def step(self, action):
        """action: Velocities in xyz and gripper_width [vx, vy, vz, vf]"""
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        action = self.clamp_action(action)

        state = self.robot.get_states()
        # Relative step in xyz
        state.end_effector.orientation =  self.cfg.sawyer_gripper.init_state.end_effector.orientation
        state.end_effector.position[0] += action[0] * self.dt
        state.end_effector.position[1] +=  action[1] * self.dt
        state.end_effector.position[2] +=  action[2] * self.dt
        state.gripper_width += action[3] * self.dt
        state["gripper_force"] = 35
        self.robot.set_actions(state)

        # TODO: Check if still required
        # Perform more steps in simulation than querying the model 
        # (Gives time to reach the joint position)
        for _ in range(self.simulation_frequency//self.action_frequency):
            p.stepSimulation()
        
        if self.cfg.tacto.visualize_gui:
            self.update_tacto_gui()

        observation = self.get_current_state()
        done, success  = self.get_termination()
        reward = 0

        if self.locked and self.get_door_handle_angle() > 0.5:
            self.unlock_door()

        info = {"door_handle_position": self.get_door_handle_position(), "success": success} 
        return observation, reward, done, info

    def lock_door(self):
        self.locked = True
        p.setJointMotorControl2(self.door.id, 0, controlMode=p.POSITION_CONTROL, 
                                targetPosition=0, force=float('inf'))

    def unlock_door(self):
        self.logger.info("Door unlocked")
        self.locked = False
        p.setJointMotorControl2(self.door.id, 0, controlMode=p.POSITION_CONTROL, 
                                targetPosition=0, force=0)

    def load_door(self):
        door_urdf_path = pkg_path("envs/" + self.cfg.objects.door.urdf_path)
        door_obj_cfg = {**self.cfg.objects.door, "urdf_path": door_urdf_path}
        self.door = px.Robot(**door_obj_cfg, physics_client=self.physics_client)
        self.digits.add_body(self.door)

    def load_objects(self):
        # Initialize digit and robot
        self.digits = tacto.Sensor(**self.cfg.tacto)
        if "envs" not in self.cfg.sawyer_gripper.robot_params.urdf_path:
            robot_urdf_path = pkg_path("envs/" + self.cfg.sawyer_gripper.robot_params.urdf_path)
            self.cfg.sawyer_gripper.robot_params.urdf_path = robot_urdf_path
        self.robot = SawyerGripper(**self.cfg.sawyer_gripper, physics_client=self.physics_client)
        self.digits.add_camera(self.robot.id, self.robot.digit_links)

        # Load objects
        self.table = px.Body(**self.cfg.objects.table, physics_client=self.physics_client)
        self.load_door()
    
    def get_termination(self):
        done, success = False, False   
        if self.get_door_angle() > 0.8:
            done, success = True, True
        return done, success

    def init_cameras(self):
        self.cameras = {}
        for key, value in self.cfg.cameras.items():
            self.cameras[key] = StaticCamera(**value, cid=self.physics_client._id)

    def render(self, mode='view_1'):
        """
         Return rgb and depth images from the cameras declared in the yaml files
         modes: 'view_1' and 'view_2' return the respective viewpoint
                'all_views' return both viewpoints
        """
        if mode == "view_1":
            return self.cameras['view_1'].render()
        elif mode == "view_2":
            return self.cameras['view_2'].render()
        elif mode == "all_views":    
            return {'view_1': self.cameras['view_1'].render(),
                    'view_2': self.cameras['view_2'].render()}

    def update_tacto_gui(self):
        color, depth = self.digits.render()
        self.digits.updateGUI(color, depth)

    def get_forces(self):
        forces = []
        for cam in self.digits.cameras:
            force_dict = self.digits.get_force(cam)
            av_force = 0
            if len(force_dict) > 0:
                for k, v in force_dict.items():
                    av_force += v
                av_force /= len(force_dict)
            scaled_force = av_force/100
            forces.append(scaled_force)
        return np.array(forces)

    def get_digit_depth(self):
        color, depth = self.digits.render()
        return depth

    def get_current_state(self):
        observation = {}
        observation["position"] = self.get_end_effector_position()
        observation["gripper_width"] = self.get_gripper_width()
        return observation

    def get_gripper_width(self):
        return np.array([self.robot.get_states().gripper_width])

    def get_end_effector_position(self):
        end_effector_position = self.robot.get_states().end_effector.position
        end_effector_position[2] -= 0.125
        return end_effector_position

    def get_door_angle(self):
        return self.door.get_joint_states().joint_position[0]

    def get_door_handle_angle(self):
        return self.door.get_joint_states().joint_position[1]

    def get_door_handle_position(self):
        return self.door.get_link_states().link_world_position[1]

    def close(self):
        p.disconnect(self.physics_client.id)


# Custom wrappers
class TransformObservation(ObservationWrapper):
    def __init__(self, env=None, with_force = False, with_tactile_sensor = False,
                 with_gripper_width=False, relative = True, with_noise=False, 
                 normalize_tactile=False):
        super(TransformObservation, self).__init__(env)
        self.with_tactile_sensor = with_tactile_sensor
        self.with_gripper_width = with_gripper_width
        self.with_noise = with_noise
        self.with_force = with_force
        self.relative = relative
        self.observation_space = self.get_observation_space()
        self.normalize_tactile = normalize_tactile

    def get_observation_space(self):
        observation_space = {}
        observation_space["position"] = spaces.Box(
                low=np.array([0.3, -0.85, 0]), high=np.array([0.85, 0.85, 0.8]))
        if self.with_gripper_width:
            observation_space["gripper_width"] = spaces.Box(low=0.03, high=0.11, shape=(1,))
        if self.with_tactile_sensor:
            t_width, t_height = self.env.cfg.tacto.width, self.env.cfg.tacto.height
            observation_space["tactile_sensor"] = spaces.Tuple(
                tuple([spaces.Box(low=-np.ones((t_height,t_width)), high=np.ones((t_height, t_width)))
                       for _ in range(2)]))
        if self.with_force:
            observation_space["force"] = spaces.Box(
                low=np.array([0, 0]), high=np.array([2.0, 2.0]))
        return SpaceDict(observation_space)

    def observation(self, obs):
        if self.with_force:
            obs["force"] = self.env.get_forces()
        if not self.with_gripper_width:
            del obs["gripper_width"]
        if self.with_tactile_sensor:
            obs["tactile_sensor"] = self.env.get_digit_depth()
            if self.normalize_tactile:
                # mean and std obtained experimentally
                mean = -0.0021
                std = 0.0073
                obs["tactile_sensor"][0] = (obs["tactile_sensor"][0] - mean)/std
                obs["tactile_sensor"][1] = (obs["tactile_sensor"][1] - mean)/std

        if self.relative:
            state_target = self.env.get_door_handle_position()
            state_target[2] += 0.020 
            if self.with_noise:
                state_target +=  self.env.target_noise
            obs["position"] -= state_target
            if self.with_gripper_width:
                obs["gripper_width"] -= 0.075
            if self.with_force:
                obs["force"] -= np.array([ 0.865, 0.865])
        return obs

class TransformAction(ActionWrapper):
    def __init__(self, env=None, with_gripper_width=False):
        super(TransformAction, self).__init__(env)
        self.with_gripper_width = with_gripper_width

        if with_gripper_width:
            self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4)) # vel_x, vel_y, vel_z, vel_joint
        else:
            self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3)) # vel_x, vel_y, vel_z

    def action(self, action):
        if self.with_gripper_width:
            return action
        else:
            action = np.append(action, -0.002/self.env.dt) 
            return action

def make_env_sawyer_door_v0(cfg):
    env = gym.make('gym_tacto:sawyer-door-v0', cfg=cfg)
    env = TransformObservation(env, **cfg.observation)
    env = TransformAction(env, **cfg.action)
    return env