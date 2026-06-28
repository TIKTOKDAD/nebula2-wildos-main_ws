from enum import Enum, auto
import numpy as np
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from .pcl_utils import create_colored_pointcloud2
from .camera_data import Camera, get_cam_intrinsics_and_extrinsics
from .bbox_generator import BoundingBoxGenerator

class ParticleDistribution(Enum):
    UNIFORM = auto()
    GAUSSIAN_HYP = auto()
    UNIFORM_HYP = auto()


class ParticleGenerator:
    """
    Generate a random set of particles within the camera frustum extending
    from the camera to the object location within the bounding box.
    """
    default_config = {
        "num_particles": 1000, # Number of particles to generate
        "depth_range": [1.0, 100.0],  # Depth range in meters
        "add_odom_drift": False, # Whether to add odometry drift to the camera pose
        "t_drift_std": 1.0,  # Standard deviation of translation drift in meters
        "r_drift_std": 2.0,  # Standard deviation of rotation drift in degrees
        "hypothesis_std": 5.0,  # Standard deviation from the depth hypothesis for sampling particles
    }
    
    def __init__(self, config: OmegaConf):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        self.num_particles = config.num_particles
        self.depth_range = np.array(config.depth_range, dtype=np.float64)
        self.add_odom_drift_flag = config.add_odom_drift
        self.t_drift_std = config.t_drift_std
        self.r_drift_std = np.deg2rad(config.r_drift_std)
        self.hypothesis_std = config.hypothesis_std

    def add_odom_drift(self, R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Add odometry drift to the camera pose.
        """
        # Add noise to the translation
        t_noise = np.random.normal(0, self.t_drift_std, size=(3,))
        t_wc_drifted = t_wc + t_noise

        # Add noise to the rotation
        R_wc_euler = R.from_matrix(R_wc).as_euler('xyz')
        r_noise = np.array([np.random.normal(0, self.r_drift_std), 0, np.random.normal(0, self.r_drift_std)])
        R_wc_drifted = R.from_euler('xyz', R_wc_euler + r_noise).as_matrix()

        return R_wc_drifted, t_wc_drifted
    
    def get_gaussian_samples(
            self, camera: Camera, depth_hypothesis: dict, num_samples: int, distribtion: ParticleDistribution
        ) -> np.ndarray:
        """
        Generate Gaussian samples based on the depth hypothesis.
        """
        cam_data = get_cam_intrinsics_and_extrinsics(camera)
        direction = camera.box_ray.direction

        if len(depth_hypothesis['location']) == 2:
            loc_2d = depth_hypothesis['location']  # (x, y)
            dist_2d = np.linalg.norm(loc_2d - np.array([cam_data.t[0], cam_data.t[1]]))

            cos_angle_z = direction[2]
            cos_angle_xy = np.sqrt(1-cos_angle_z**2)
            dist_3d = dist_2d / cos_angle_xy

        elif len(depth_hypothesis['location']) == 3:
            loc_3d = depth_hypothesis['location'] # (x, y, z)
            dist_3d = np.linalg.norm(loc_3d - cam_data.t)  # Distance from camera to object in 3D space

        else:
            raise ValueError("Depth hypothesis location must be 2D or 3D.")

        # dist_3d is the distance from the camera to the object in 3D space. Convert to Z-depth
        cam_forward_world = cam_data.R @ np.array([0, 0, 1])  # Forward direction in world frame
        cam_forward_world /= np.linalg.norm(cam_forward_world)  # Normalize
        cos_z_cam = np.dot(direction, cam_forward_world)  # Cosine of angle between camera forward and ray direction
        depth_3d = dist_3d * cos_z_cam  # Convert to Z-depth in camera frame

        # Generate samples around the depth hypothesis
        if distribtion == ParticleDistribution.GAUSSIAN_HYP:
            samples = np.random.normal(depth_3d, self.hypothesis_std, size=(num_samples,))
        elif distribtion == ParticleDistribution.UNIFORM_HYP:
            # Uniform samples around the depth hypothesis
            lower_bound = max(depth_3d - self.hypothesis_std, self.depth_range[0])
            upper_bound = min(depth_3d + self.hypothesis_std, self.depth_range[1])
            samples = np.random.uniform(lower_bound, upper_bound, size=(num_samples,))
        else:
            raise ValueError("Invalid distribution type.")
        
        samples = np.clip(samples, self.depth_range[0], self.depth_range[1])  # Ensure samples are within the depth range
        return samples

    def generate_particles(
            self,
            camera: Camera,
            depth_hypothesis: dict = None,
            distribution: ParticleDistribution = ParticleDistribution.UNIFORM,
            color: np.ndarray = None,
            use_mask: bool=False,
            pcl_frame_id: str = 'world'
        ) -> Camera:
        """
        Select a random set of points within the bounding box of the object,
        and project to a random depth within the camera frustum.
        The points are then transformed to the world frame.
        """
        K = np.array(camera.camera_info.k).reshape(3, 3)
        K_inv = np.linalg.inv(K)

        if use_mask and camera.object_mask is not None:
            # Use the object mask to generate points
            mask = camera.object_mask
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                raise ValueError("Object mask is empty, cannot generate particles.")
            
            # Randomly select points from the mask
            indices = np.random.choice(len(x_indices), self.num_particles, replace=True)
            x = x_indices[indices]
            y = y_indices[indices]
            
        else:
            bounding_box = camera.bounding_box
            x_min = bounding_box[0]
            x_max = bounding_box[2]
            y_min = bounding_box[1]
            y_max = bounding_box[3]

            # Generate random points within the bounding box
            x = np.random.uniform(x_min, x_max, self.num_particles)
            y = np.random.uniform(y_min, y_max, self.num_particles)

        # Generate random depths
        if depth_hypothesis is None or distribution == ParticleDistribution.UNIFORM:
            z = np.random.uniform(self.depth_range[0], self.depth_range[1], self.num_particles)   # shape (N,)
        elif distribution == ParticleDistribution.GAUSSIAN_HYP or distribution == ParticleDistribution.UNIFORM_HYP:
            z = self.get_gaussian_samples(camera, depth_hypothesis, self.num_particles, distribution)
        else:
            raise ValueError("Invalid distribution type.")

        points_2d = np.vstack((x, y, np.ones(self.num_particles)))  # shape (3, N)
        points_3d = (K_inv @ points_2d) * z.reshape(1, -1)  # shape (3, N)

        # Transform the points to the world frame
        cam_data = get_cam_intrinsics_and_extrinsics(camera)
        R_wc = cam_data.R
        t_wc = cam_data.t
        if self.add_odom_drift_flag:
            R_wc, t_wc = self.add_odom_drift(R_wc, t_wc)

            # Change camera ray direction to match the drifted pose
            camera = BoundingBoxGenerator.generate_ray_from_bbox(camera, R_wc, t_wc)
        
        points_3d_world = (R_wc @ points_3d) + t_wc.reshape(3, 1)  # shape (3, N)
        points_3d_world = points_3d_world.T  # shape (N, 3)

        # Create a PointCloud2 message
        if color is None:
            color = np.array([255, 255, 255], dtype=np.uint8)
        pcl_msg = create_colored_pointcloud2(
            points_3d_world,
            frame_id=pcl_frame_id,
            color=color
        )

        camera.points = pcl_msg

        return camera
