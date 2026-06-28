from dataclasses import dataclass, field

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Ray:
    """
    Represents a ray in 3D space.
    Contains the origin and direction of the ray.
    The origin is the camera location, and the direction is the unit vector pointing in the direction of the ray.
    """
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        if self.origin.shape != (3,):
            raise ValueError("Origin must be a 3D vector.")
        if self.direction.shape != (3,):
            raise ValueError("Direction must be a 3D vector.")
        if not np.isclose(np.linalg.norm(self.direction), 1.0):
            raise ValueError("Direction must be a unit vector.")
        
    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the distance from the ray to a set of points.
        Points should be a 2D array of shape (N, 3).
        Returns an array of distances of shape (N,).
        """
        
        if points.shape[1] != 3:
            raise ValueError("Points must be a 2D array of shape (N, 3).")
        
        # Calculate the vector from the ray origin to the points
        vec_to_points = points - self.origin.reshape(1, 3)
        
        # Project the vector onto the ray direction
        projection_length = np.dot(vec_to_points, self.direction.reshape(3, 1)) # (N, 3) @ (3, 1) -> (N, 1)
        projection = projection_length * self.direction.reshape(1, 3)  # (N, 1) * (1, 3) -> (N, 3)
        projection = projection + self.origin.reshape(1, 3)  # Add the ray origin to the projection
        
        # Calculate the distance from the points to the ray
        distance_vec = points - projection  # (N, 3) - (N, 3) -> (N, 3)
        distances = np.linalg.norm(distance_vec, axis=1).reshape(-1, 1)

        # Make the points that are behind the ray origin have infinite distance
        behind_origin = projection_length <= 0
        distances[behind_origin] = np.inf
        
        return distances
    
    @staticmethod
    def get_2d_ray_intersection(ray1: 'Ray', ray2: 'Ray') -> np.ndarray:
        """
        Calculate the intersection point of two rays in 2D.
        The rays are defined in the XY plane (Z=0).
        Returns the intersection point as a 2D vector [x, y].
        If the rays are parallel, returns None.
        """
        # Convert the rays to 2D
        p1 = ray1.origin[:2]
        d1 = ray1.direction[:2]
        d1 /= np.linalg.norm(d1)

        p2 = ray2.origin[:2]
        d2 = ray2.direction[:2]
        d2 /= np.linalg.norm(d2)

        # Solve for the intersection point
        A = np.array([d1, -d2]).T
        b = p2 - p1

        if np.linalg.det(A) == 0:
            print("Least squares solution failed, rays may be parallel or insufficient data.")
            return None
        t = np.linalg.solve(A, b)

        # Check if the intersection point is in front of both rays
        if t[0] < 0 or t[1] < 0:
            print("Intersection point is behind the origin of one or both rays.")
            return None

        return p1 + t[0] * d1, t
    
    @staticmethod
    def get_multiple_2d_ray_intersections(rays: list['Ray']) -> np.ndarray:
        """
        Calculate the least squares estimate of the intersection point of multiple rays in 2D.
        The rays are defined in the XY plane (Z=0).
        Returns the intersection point as a 2D vector [x, y].
        If the rays intersect behind the origin, returns None.
        """
        if len(rays) < 2:
            raise ValueError("At least two rays are required to calculate an intersection.")

        A = []
        b = []

        for ray in rays:
            p = ray.origin[:2]
            d = ray.direction[:2]
            d = d / np.linalg.norm(d)

            # Construct the projection matrix to compute orthogonal component
            I = np.eye(2)
            P = I - np.outer(d, d)  # Projects onto the plane orthogonal to d

            A.append(P)
            b.append(P @ p)

        A = np.vstack(A)  # Shape: (2N, 2)
        b = np.concatenate(b)  # Shape: (2N,)

        # Solve the least squares problem
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            print("Least squares solution failed, rays may be parallel or insufficient data.")
            return None
        
        #TODO: Check if the intersection point is in front of all rays

        return x
    
    @staticmethod
    def get_multiple_3d_ray_intersections(rays: list['Ray']) -> np.ndarray:
        """
        Calculate the least squares estimate of the intersection point of multiple rays in 3D.
        Returns the intersection point as a 3D vector [x, y, z].
        """
        if len(rays) < 2:
            raise ValueError("At least two rays are required to calculate an intersection.")

        A = []
        b = []

        for ray in rays:
            p = ray.origin[:3]
            d = ray.direction[:3]
            d = d / np.linalg.norm(d)

            # Projection matrix that removes the component along d
            I = np.eye(3)
            P = I - np.outer(d, d)

            A.append(P)
            b.append(P @ p)

        A = np.vstack(A)  # Shape: (3N, 3)
        b = np.concatenate(b)  # Shape: (3N,)

        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        return x

            

@dataclass
class CameraPose:
    """
    Pose information for the camera.
    Contains the translation and rotation of the camera in the `world` frame.
    Rotation is represented as a quaternion.
    """

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))

    def __post_init__(self):
        if self.translation.shape != (3,):
            raise ValueError("Translation must be a 3D vector.")
        if self.rotation.shape != (4,):
            raise ValueError("Rotation must be a quaternion.")
        if not np.isclose(np.linalg.norm(self.rotation), 1.0):
            raise ValueError("Rotation must be a unit quaternion.")
        

@dataclass
class Camera:
    """
    Camera information.
    Contains the camera intrinsics, camera pose, and 2D bounding box of the object in the camera frame.
    Bounding box is represented as a 4D vector [x_min, y_min, x_max, y_max].
    """
    camera_info: CameraInfo
    camera_tf: TransformStamped
    bounding_box: np.ndarray = field(default_factory=lambda: np.zeros(4))
    object_mask: np.ndarray = None # Optional mask for the object in the image
    box_ray: Ray = field(default_factory=lambda: Ray(np.zeros(3), np.array([0.0, 0.0, 1.0])))
    image : Image = field(default_factory=lambda: Image())
    points: PointCloud2 = field(default_factory=lambda: PointCloud2())


@dataclass
class CamIntrinsicsAndExtrinsics:
    """
    Camera intrinsics and extrinsics.
    Contains the camera intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        if self.K.shape != (3, 3):
            raise ValueError("K must be a 3x3 matrix.")
        if self.R.shape != (3, 3):
            raise ValueError("R must be a 3x3 matrix.")
        if self.t.shape != (3,):
            raise ValueError("t must be a 3D vector.")
        

def get_extrinsics_from_tfmsg(tf_msg: TransformStamped) -> CamIntrinsicsAndExtrinsics:
    """
    Extract camera extrinsics from a TransformStamped message.
    Returns a CamIntrinsicsAndExtrinsics object containing R and t.
    """
    R_wc_quat = np.array([
        tf_msg.transform.rotation.x,
        tf_msg.transform.rotation.y,
        tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w
    ])
    R_wc = R.from_quat(R_wc_quat).as_matrix()
    
    t_wc = np.array([
        tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z
    ])
    
    return CamIntrinsicsAndExtrinsics(K=np.eye(3), R=R_wc, t=t_wc)


def get_cam_intrinsics_and_extrinsics(camera: Camera) -> CamIntrinsicsAndExtrinsics:
    """
    Extract camera intrinsics and extrinsics from the Camera object.
    Returns a CamIntrinsicsAndExtrinsics object containing K, R, and t.
    """
    K = np.array(camera.camera_info.k).reshape(3, 3)
    cam_data = get_extrinsics_from_tfmsg(camera.camera_tf)
    cam_data.K = K
    
    return cam_data