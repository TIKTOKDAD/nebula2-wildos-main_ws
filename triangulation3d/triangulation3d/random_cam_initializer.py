import numpy as np
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped

from .camera_data import CameraPose


class RandomCameraInitialization:
    """
    Randomly initialize the camera poses around the object location.
    The camera poses are generated based on the object location in the world frame.
    The camera poses are generated in a circular pattern around the object location and 
    in a random distance and height from the object location.
    """
    default_config = {
        "num_cameras": 10,  # Number of cameras to generate
        "cam_distance_range": [30.0, 70.0],  # Distance range from the object in meters
        "cam_height_range": [-2.0, -5.0],  # Height range from the object in meters
        "cam_spread_angle": 90.0,  # Spread angle of the cameras in degrees
        "ground_aligned": True,  # If True, cameras are aligned with the ground plane
    }

    def __init__(self, object_loc_world: np.ndarray, config: OmegaConf):
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        self.cam_info = self.load_camera_info()
        self.object_loc_world = object_loc_world

        self.num_cameras = config.num_cameras
        self.cam_distance_range = np.array(config.cam_distance_range, dtype=np.float64)
        self.cam_height_range = np.array(config.cam_height_range, dtype=np.float64)
        self.cam_spread_angle = np.deg2rad(config.cam_spread_angle)  # Convert to radians
        self.ground_aligned = config.ground_aligned

        # Change of basis matrix from camera to world frame
        # This matrix is used to transform the camera coordinates to world coordinates.
        # Camera frame: X axis points to the right, Y axis points down, Z axis points forward.
        # World frame: X axis points to the right, Y axis points forward, Z axis points up.
        self.R_CAM_TO_WORLD = np.array(
            [[1, 0, 0],
             [0, 0, 1],
             [0, -1, 0]]
        )

    def load_camera_info(self) -> CameraInfo:
        """
        Load the camera intrinsics and return them as a CameraInfo message.
        This function assumes a pinhole camera model.
        """
        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        fov = 90 # field of view in degrees

        width = 640
        height = 480

        # Calculate the intrinsic matrix
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        # Create the CameraInfo message
        camera_info = CameraInfo()
        camera_info.header.frame_id = 'camera'
        camera_info.width = width
        camera_info.height = height
        camera_info.k = K.flatten().tolist()
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        camera_info.r = np.eye(3).flatten().tolist()  # Rectification matrix
        camera_info.p = np.concatenate((K, np.zeros((3, 1))), axis=1).flatten().tolist()  # Projection matrix

        return camera_info
    
    @staticmethod
    def campose_to_transform(
            cam_pose: CameraPose, world_frame_id: str='world', child_frame_id: str='camera'
        ) -> TransformStamped:
        """
        Convert a CameraPose object to a TransformStamped message.
        """
        t = TransformStamped()
        t.header.frame_id = world_frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = cam_pose.translation[0]
        t.transform.translation.y = cam_pose.translation[1]
        t.transform.translation.z = cam_pose.translation[2]
        t.transform.rotation.x = cam_pose.rotation[0]
        t.transform.rotation.y = cam_pose.rotation[1]
        t.transform.rotation.z = cam_pose.rotation[2]
        t.transform.rotation.w = cam_pose.rotation[3]

        return t

    def generate_camera(
            self, object_loc_world: np.ndarray, ground_aligned: bool
        ) -> CameraPose:
        """
        Generate a random camera pose around the object location.
        The camera is placed at a random distance and height from the object.
        Each camera can see the object. If `ground_aligned` is set to False, 
        object location in the camera frame is randomly sampled; else it is set to center of the image.
        The rotation of the camera is calculated based on the object location.
        """
        cam_pose = CameraPose()

        # Randomly sample the position of the camera
        distance = np.random.uniform(self.cam_distance_range[0], self.cam_distance_range[1])
        height = np.random.uniform(self.cam_height_range[0], self.cam_height_range[1])
        angle = np.random.uniform(0, self.cam_spread_angle)
        cam_pose.translation[0] = object_loc_world[0] + distance * np.cos(angle)
        cam_pose.translation[1] = object_loc_world[1] + distance * np.sin(angle)
        cam_pose.translation[2] = object_loc_world[2] + height

        # Randomly sample the object location in the camera frame
        img_w = self.cam_info.width
        img_h = self.cam_info.height
        object_loc_camera = np.array([
            np.random.uniform(0, img_w) if not ground_aligned else img_w / 2,
            np.random.uniform(0, img_h) if not ground_aligned else img_h / 2,
            1.0
        ])

        # vector from camera to object in world frame
        twc = cam_pose.translation
        obj_vec_world = object_loc_world - twc
        obj_vec_world /= np.linalg.norm(obj_vec_world)

        if not ground_aligned:
            # Calculate the rotation of the camera based on the object location
            K = np.array(self.cam_info.k).reshape(3, 3)
            K_inv = np.linalg.inv(K)
            obj_depth = np.linalg.norm(object_loc_world - twc)
            obj_cam = (K_inv @ object_loc_camera.T) * obj_depth

            obj_ray = obj_cam / np.linalg.norm(obj_cam)
            obj_ray_world = self.R_CAM_TO_WORLD @ obj_ray
            obj_ray_world /= np.linalg.norm(obj_ray_world)
            
            # Rwc.Pc + twc = obj_loc. Solve for Rwc. Multiple solutions exist.
            r_align, _ = R.align_vectors(
                obj_vec_world.reshape(1, 3),
                obj_ray_world.reshape(1, 3)
            )
            cam_pose.rotation = R.from_matrix(
                r_align.as_matrix() @ self.R_CAM_TO_WORLD
            ).as_quat()

        else:
            # If ground aligned, we assume the camera is looking at the object directly
            # and the camera's x-axis is parallel to the ground plane (xy).
            cam_z_axis = obj_vec_world
            cam_y_axis = np.array([0, 0, -1])

            # Compute orthonormal basis
            x_axis = np.cross(cam_y_axis, cam_z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(cam_z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)

            # Construct rotation matrix
            R_wc = np.stack([x_axis, y_axis, cam_z_axis], axis=1)
            cam_pose.rotation = R.from_matrix(R_wc).as_quat()

        return cam_pose
    
    def generate_cameras(self, num_cameras: int) -> list[TransformStamped]:
        """
        Generate a list of random camera poses around the object location.
        """
        cameras = []

        for cam_idx in range(num_cameras):
            cam_pose = self.generate_camera(self.object_loc_world, self.ground_aligned)
            t = RandomCameraInitialization.campose_to_transform(
                cam_pose, world_frame_id='world', child_frame_id=f'camera_{cam_idx}'
            )
            cameras.append(t)

        return cameras
