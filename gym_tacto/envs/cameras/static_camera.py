import numpy as np
import pybullet as p
from gym_tacto.envs.cameras.camera import Camera


class StaticCamera(Camera):
    def __init__(self, fov, aspect, nearval, farval, width, height, look_at, look_from, cid, name):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self.nearval = nearval
        self.farval = farval
        self.width = width
        self.height = height
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=[0.0, 0.0, 1.0]
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self.nearval, farVal=self.farval
        )
        self.cid = cid
        self.name = name

    def set_position_from_gui(self):
        info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=[0.0, 0.0, 1.0]
        )
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def render(self):
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)
        return rgb_img, depth_img
