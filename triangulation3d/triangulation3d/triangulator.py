from enum import Enum, auto
import numpy as np
from omegaconf import OmegaConf

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

from .camera_data import Camera
from .pcl_utils import combine_pointcloud2


class TriangulationMethod(Enum):
    WEIGHTED = auto()
    MINIMUM = auto()


class Triangulator:
    default_config = {
        "triangulation_uncertainty": 10.0,  # Uncertainty in triangulation, radius of the sphere around the triangulated point
        "triangulation_method": "WEIGHTED"  # Triangulation method to use
    }

    def __init__(self, config: OmegaConf = None):
        """
        Initialize the triangulator.
        This class is responsible for triangulating the object location from multiple camera frames.
        """
        config = OmegaConf.create() if config is None else config
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        self.triangulation_uncertainty = config.triangulation_uncertainty
        self.triangulation_method = TriangulationMethod[config.triangulation_method]

    def triangulate(self, cameras: list[Camera]) -> np.ndarray:
        """
        Triangulate the object location from projected particles.
        For each 3D particle, calculate the weight based on the distance to the
        bounding box ray from each camera. Calculate the weighted average of the particles
        to get the final triangulated position.
        """

        combined_pcl_msg = self.combine_points(cameras)
        points3d = point_cloud2.read_points(combined_pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
        points3d = np.array([np.array(list(pt)) for pt in points3d])

        point_weights = np.zeros(points3d.shape[0]).reshape(-1, 1)
        point_distances = np.zeros(points3d.shape[0]).reshape(-1, 1)
        for camera in cameras:
            box_ray = camera.box_ray
            distances = box_ray.distance_to_points(points3d)
            point_distances += distances
            # Inverse distance weighting
            weights = 1.0 / (distances + 1e-3)
            # convert nan weights to 0
            weights[np.isnan(weights)] = 0.0
            point_weights += weights

        # point_weights = 1.0 / (point_distances + 1e-3)

        point_weights /= np.sum(point_weights)

        if self.triangulation_method == TriangulationMethod.WEIGHTED:
            triangulated_position = np.sum(points3d * point_weights, axis=0)
        elif self.triangulation_method == TriangulationMethod.MINIMUM:
            triangulated_position = points3d[np.argmin(point_distances)].astype(np.float64)
        else:
            raise ValueError(f"Unknown triangulation method: {self.triangulation_method}")

        return triangulated_position

    def combine_points(self, cameras: list[Camera], pcl_frame_id: str="world") -> PointCloud2:
        """
        Combine the points from all the cameras into a single PointCloud2 message.
        """
        combined_pcl_msg = PointCloud2()
        clouds = []
        for camera in cameras:
            clouds.append(camera.points)
        combined_pcl_msg = combine_pointcloud2(clouds)
        combined_pcl_msg.header.frame_id = pcl_frame_id
        
        return combined_pcl_msg
    
    def get_triangulated_marker(self, triangulated_position: np.ndarray, marker_frame_id: str='world') -> Marker:
        """
        Return a marker for the triangulated position.
        """
        marker = Marker()
        marker.header.frame_id = marker_frame_id
        marker.ns = 'triangulated_position'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0)
        marker.pose.position.x = triangulated_position[0]
        marker.pose.position.y = triangulated_position[1]
        marker.pose.position.z = triangulated_position[2]
        marker.scale.x = 1.0 * self.triangulation_uncertainty
        marker.scale.y = 1.0 * self.triangulation_uncertainty
        marker.scale.z = 1.0 * self.triangulation_uncertainty
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker
