import numpy as np
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from .camera_data import Camera, Ray, get_cam_intrinsics_and_extrinsics

class BoundingBoxGenerator:
    """
    Generate a bounding box around the object location in
    all the camera frames. Assumption: The object is a cube of side 1m.
    The bounding box is generated in the world frame and then transformed to the camera frame.
    """
    default_config = {
        "noise_mean": 0.0,  # mean of the noise in pixels
        "noise_std": 15.0   # standard deviation of the noise in pixels
    }

    def __init__(self, object_loc_world: np.ndarray, object_size: float, config: OmegaConf):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        self.object_loc_world = object_loc_world
        self.cube_side_half = 0.5 * object_size  # half the side length of the cube
        cube_vertices = np.array([
            [-self.cube_side_half, -self.cube_side_half, -self.cube_side_half],
            [self.cube_side_half, -self.cube_side_half, -self.cube_side_half],
            [self.cube_side_half, self.cube_side_half, -self.cube_side_half],
            [-self.cube_side_half, self.cube_side_half, -self.cube_side_half],
            [-self.cube_side_half, -self.cube_side_half, self.cube_side_half],
            [self.cube_side_half, -self.cube_side_half, self.cube_side_half],
            [self.cube_side_half, self.cube_side_half, self.cube_side_half],
            [-self.cube_side_half, self.cube_side_half, self.cube_side_half]
        ]).T
        # Transform the cube vertices to the world frame
        self.cube_vertices_world = cube_vertices + object_loc_world.reshape(3, 1)
        
        # noise level for the bounding box detections
        self.noise_mean = config.noise_mean
        self.noise_std = config.noise_std

    def generate_bounding_box(self, cam: Camera) -> Camera:
        """
        Transform the 2D bounding box pixel coordinates.
        Transform the cube vertices from the world frame to the camera frame,
        and take the bounds along the camera axes.
        """

        # Transform the cube vertices to the camera frame
        cam_data = get_cam_intrinsics_and_extrinsics(cam)
        R_wc = cam_data.R
        t_wc = cam_data.t
        K = cam_data.K

        cube_vertices_camera = R_wc.T @ (self.cube_vertices_world - t_wc.reshape(3, 1))

        # Check if the cube vertices are in front of the camera
        if np.any(cube_vertices_camera[2, :] < 0):
            raise ValueError("Some cube vertices are behind the camera. Cannot generate bounding box.")

        cube_vertices_pixels = K @ cube_vertices_camera
        cube_vertices_pixels /= cube_vertices_pixels[2, :]
        cube_vertices_pixels = cube_vertices_pixels[:2, :]

        # Get the bounding box in pixel coordinates
        x_min = np.min(cube_vertices_pixels[0, :])
        x_max = np.max(cube_vertices_pixels[0, :])
        y_min = np.min(cube_vertices_pixels[1, :])
        y_max = np.max(cube_vertices_pixels[1, :])

        # Add noise to the bounding box coordinates
        x_min += np.random.normal(self.noise_mean, self.noise_std)
        x_max += np.random.normal(self.noise_mean, self.noise_std)
        y_min += np.random.normal(self.noise_mean, self.noise_std)
        y_max += np.random.normal(self.noise_mean, self.noise_std)

        # Ensure the bounding box is within the image bounds
        x_min = max(0, x_min)
        x_max = min(cam.camera_info.width-1, x_max)
        y_min = max(0, y_min)
        y_max = min(cam.camera_info.height-1, y_max)

        bounding_box = np.array([x_min, y_min, x_max, y_max])
        cam.bounding_box = bounding_box

        # Create a Ray object for the bounding box
        cam = BoundingBoxGenerator.generate_ray_from_bbox(cam)

        return cam
    
    @staticmethod
    def generate_ray_from_bbox(cam: Camera, R_wc: np.ndarray=None, t_wc: np.ndarray=None) -> Camera:
        """
        Generate a ray in the bounding box direction in the camera frame.
        The ray is defined by the camera principal point and the center of the bounding box.
        """
        # Get the bounding box in pixel coordinates
        x_min, y_min, x_max, y_max = cam.bounding_box

        cam_data = get_cam_intrinsics_and_extrinsics(cam)
        K = cam_data.K
        
        if R_wc is None or t_wc is None:
            R_wc = cam_data.R
            t_wc = cam_data.t

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        box_center_pixel = np.array([cx, cy, 1.0])
        K_inv = np.linalg.inv(K)
        box_center_cam = K_inv @ box_center_pixel.reshape(3, 1)

        ray_origin_cam = np.zeros(3).reshape(3,1) # Camera origin in camera frame
        ray_direction_cam = box_center_cam / np.linalg.norm(box_center_cam)
        
        ray_origin_world = R_wc @ ray_origin_cam + t_wc.reshape(3, 1)
        ray_direction_world = R_wc @ ray_direction_cam
        box_ray = Ray(
            origin=ray_origin_world.flatten(),
            direction=ray_direction_world.flatten()
        )
        cam.box_ray = box_ray

        return cam

        
