import numpy as np


class Camera:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def z_buffer_to_real_distance(self, z_buffer, far, near):
        """ Function to transform depth buffer values to distances in camera space """
        return far * near / (far - (far - near) * z_buffer)

    def process_rgbd(self, obs, nearval, farval):
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = obs
        rgb = np.reshape(rgbPixels, (height, width, 4))
        rgb_img = rgb[:, :, :3]
        depth_buffer = np.reshape(depthPixels, [height, width])
        depth = self.z_buffer_to_real_distance(z_buffer=depth_buffer, far=farval, near=nearval)
        return rgb_img, depth
