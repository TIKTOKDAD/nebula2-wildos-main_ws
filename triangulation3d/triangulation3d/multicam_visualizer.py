from copy import deepcopy
from enum import Enum, auto
import numpy as np
import cv2
from cv_bridge import CvBridge
from omegaconf import OmegaConf

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from builtin_interfaces.msg import Duration

import rclpy
from rclpy.node import Node

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .pcl_utils import combine_pointcloud2
from .random_cam_initializer import RandomCameraInitialization
from .bbox_generator import BoundingBoxGenerator
from .particle_generator import ParticleGenerator, ParticleDistribution
from .triangulator import Triangulator
from .camera_data import Camera, Ray


class HypothesisType(Enum):
    """ Enum for the type of hypothesis to use for triangulation. """
    NONE = auto()  # No hypothesis
    DEPTH_2D = auto()  # 2D ray intersection hypothesis
    DEPTH_3D = auto()  # 3D ray intersection hypothesis


class VizDemo3DTriangulator(Node):
    """
    Initialize multiple cameras around an object and triangulate its location.
    This node publishes the camera info, images, and point cloud data for the triangulated object.
    It also publishes markers for the object location and the triangulated position.
    """
    default_config = {
        "object_loc_world": [0.0, 0.0, 10.0],  # Location of the object in the world frame
        "object_size": 10.0,  # Size of the object in meters (assumed to be a cube)

        "camera_initialization_config": {
            "num_cameras": 10,  # Number of cameras to generate
            "cam_distance_range": [60.0, 100.0],  # Distance range from the object in meters
            "cam_height_range": [-2.0, -5.0],  # Height range from the object in meters
            "cam_spread_angle": 6.0,  # Spread angle of the cameras in degrees
            "ground_aligned": True,  # If True, cameras are aligned with the ground plane
        },
        "bounding_box_config": {
            "noise_mean": 0.0,  # Mean of the noise in pixels
            "noise_std": 4.0,  # Standard deviation of the noise in pixels
        },
        "particle_generator_config": {},
        "triangulator_config": {
            "triangulation_uncertainty": 1.0,
            "triangulation_method": "WEIGHTED"
        },

        "hypothesis_type": "DEPTH_3D",  # Type of hypothesis to use for triangulation
        "particle_distribution": "GAUSSIAN_HYP", # "UNIFORM" or "GAUSSIAN_HYP" or "UNIFORM_HYP"
        "num_rays_for_hypothesis": -1,  # Number of rays to use for depth hypothesis (-1 for all)
    }

    def __init__(self, config: OmegaConf):
        super().__init__('demo_triangulator')
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        self.object_loc_world = np.array(config.object_loc_world, dtype=np.float64)
        self.object_size = config.object_size
        self.camera_initialization = RandomCameraInitialization(self.object_loc_world, config.camera_initialization_config)
        self.bounding_box_generator = BoundingBoxGenerator(self.object_loc_world, self.object_size, config.bounding_box_config)
        self.particle_generator = ParticleGenerator(config.particle_generator_config)
        self.particle_distribution = ParticleDistribution[config.particle_distribution]

        self.hypothesis_type = HypothesisType[config.hypothesis_type]
        self.num_rays_for_hypothesis = config.num_rays_for_hypothesis
        if self.num_rays_for_hypothesis == -1:
            self.num_rays_for_hypothesis = self.camera_initialization.num_cameras
        
        self.triangulator = Triangulator(config.triangulator_config)

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.bridge = CvBridge()
        self.initialize_cameras()
        self.publish_obj_marker()

        # triangulate the object location from the projected particles
        self.triangulated_position = self.triangulator.triangulate(self.cameras)
        self.get_logger().info(f"Triangulated position: {self.triangulated_position}")
        self.publish_triangulated_marker()

        self.publish_2d_hypothesis()
        self.publish_3d_hypothesis()

        self.camera_info_publishers = [
            self.create_publisher(
                CameraInfo,
                f'camera_{idx}/camera_info',
                10
            ) for idx in range(self.camera_initialization.num_cameras)
        ]
        self.image_publishers = [
            self.create_publisher(
                Image,
                f'camera_{idx}/image_raw',
                10
            ) for idx in range(self.camera_initialization.num_cameras)
        ]
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            'particles',
            10
        )
        self.timer = self.create_timer(0.1, self.publisher)

    def publisher(self):
        """
        Publish the data for all the cameras: camera info and image.
        """

        # Publish the camera info
        pub_time = self.get_clock().now().to_msg()

        for idx, camera in enumerate(self.cameras):
            camera_info = camera.camera_info
            camera_info.header.stamp = pub_time
            self.camera_info_publishers[idx].publish(camera_info)

            image_msg = camera.image
            image_msg.header.stamp = pub_time
            self.image_publishers[idx].publish(image_msg)

            self.get_logger().info(f"Published camera info and image for {camera.camera_tf.child_frame_id}")

        # Combine the points from all the cameras into a single PointCloud message
        combined_pcl_msg = self.triangulator.combine_points(self.cameras)
        combined_pcl_msg.header.stamp = pub_time
        self.point_cloud_publisher.publish(combined_pcl_msg)
        self.get_logger().info("Published combined point cloud message")

        # Publish the markers
        self.publish_obj_marker()
        self.publish_triangulated_marker()
        self.publish_2d_hypothesis()
        self.publish_3d_hypothesis()


    def get_2d_hypothesis(self, cameras: list[Camera]):
        """ Get the 2D ray intersection of 2 randomly selected cameras.
        """
        cam_indices = np.random.choice(
            len(cameras), size=self.num_rays_for_hypothesis, replace=False)
        
        if self.num_rays_for_hypothesis == 2:
            cam1, cam2 = cameras[cam_indices[0]], cameras[cam_indices[1]]
            loc_2d, depth_2d = Ray.get_2d_ray_intersection(cam1.box_ray, cam2.box_ray)
        else:
            # Use all cameras to get a least-squares intersection
            rays = [cameras[idx].box_ray for idx in cam_indices]
            loc_2d = Ray.get_multiple_2d_ray_intersections(rays)
            depth_2d = []
            for idx in cam_indices:
                cam = cameras[idx]
                depth = np.linalg.norm(loc_2d - np.array([cam.box_ray.origin[0], cam.box_ray.origin[1]]))
                depth_2d.append(depth)
            depth_2d = np.array(depth_2d)

        depth_hypothesis = {
            'location': loc_2d,
            'depths': depth_2d,
            'camera_indices': cam_indices
        }
        return depth_hypothesis


    def get_3d_hypothesis(self, cameras: list[Camera]):
        """ Get the 3D ray intersection of all cameras.
        """
        cam_indices = np.random.choice(
            len(cameras), size=self.num_rays_for_hypothesis, replace=False)
        
        rays = [cameras[idx].box_ray for idx in cam_indices]
        loc_3d = Ray.get_multiple_3d_ray_intersections(rays)
        dist_3d = []
        for idx in cam_indices:
            cam = cameras[idx]
            dist = np.linalg.norm(loc_3d - np.array([cam.box_ray.origin[0], cam.box_ray.origin[1], cam.box_ray.origin[2]]))
            dist_3d.append(dist)
        dist_3d = np.array(dist_3d)

        depth_hypothesis = {
            'location': loc_3d,
            'depths': dist_3d,
            'camera_indices': cam_indices
        }

        return depth_hypothesis

    def initialize_cameras(self):
        """
        Initialize the cameras and publish their transforms.
        """
        camera_tfs = self.camera_initialization.generate_cameras(
            self.camera_initialization.num_cameras
        )
        pub_time = self.get_clock().now().to_msg()

        self.cameras = []
        for idx, camera_tf in enumerate(camera_tfs):

            camera_tf.header.stamp = pub_time
            self.tf_static_broadcaster.sendTransform(camera_tf)
            self.get_logger().info(f"Published transform for {camera_tf.child_frame_id}")

            camera_info = deepcopy(self.camera_initialization.cam_info)
            camera_info.header.stamp = pub_time
            camera_info.header.frame_id = camera_tf.child_frame_id
            
            camera = Camera(
                camera_info=camera_info,
                camera_tf=camera_tf
            )

            # Generate bounding boxes and rays
            camera = self.bounding_box_generator.generate_bounding_box(camera)

            # Generate image
            img = np.zeros((camera_info.height, camera_info.width, 3), dtype=np.uint8)
            cv2.rectangle(
                img,
                (int(camera.bounding_box[0]), int(camera.bounding_box[1])),
                (int(camera.bounding_box[2]), int(camera.bounding_box[3])),
                (0, 255, 0),
                2
            )
            image_msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
            image_msg.header.stamp = pub_time
            image_msg.header.frame_id = camera_tf.child_frame_id
            camera.image = image_msg

            self.cameras.append(camera)
        
        # Get a depth hypothesis from ray intersections
        self.depth_hypothesis_2d = self.get_2d_hypothesis(self.cameras)
        self.depth_hypothesis_3d = self.get_3d_hypothesis(self.cameras)

        if self.hypothesis_type == HypothesisType.DEPTH_2D:
            depth_hypothesis = self.depth_hypothesis_2d
        elif self.hypothesis_type == HypothesisType.DEPTH_3D:
            depth_hypothesis = self.depth_hypothesis_3d
        else:
            depth_hypothesis = None

        # Generate particles for each camera
        for camera in self.cameras:
            camera = self.particle_generator.generate_particles(
                camera,
                depth_hypothesis=depth_hypothesis,
                distribution=self.particle_distribution,
                use_mask=False,
                pcl_frame_id='world'
            )

        self.get_logger().info("Initialized cameras and published static transforms!")

    def publish_obj_marker(self):
        """
        Publish a marker for the object location.
        """
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'object'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0)
        marker.pose.position.x = self.object_loc_world[0]
        marker.pose.position.y = self.object_loc_world[1]
        marker.pose.position.z = self.object_loc_world[2]
        marker.scale.x = 1.0 * self.object_size
        marker.scale.y = 1.0 * self.object_size
        marker.scale.z = 1.0 * self.object_size
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Publish the object location
        obj_pub = self.create_publisher(Marker, 'object_marker', 10)
        obj_pub.publish(marker)
        self.get_logger().info("Published object location")

    def publish_triangulated_marker(self):
        """
        Publish a marker for the triangulated position.
        """
        marker = self.triangulator.get_triangulated_marker(self.triangulated_position)
        marker.header.stamp = self.get_clock().now().to_msg()

        # Publish the triangulated position
        triangulated_pub = self.create_publisher(Marker, 'triangulated_marker', 10)
        triangulated_pub.publish(marker)
        self.get_logger().info(f"Published triangulated position at {self.triangulated_position}")

    def publish_2d_hypothesis(self):
        """
        Publish a marker for the 2D ray intersection.
        Also publish ray markers for the two cameras used.
        """
        hypothesis_2d_loc_pub = self.create_publisher(Marker, 'hypothesis_2d_location', 10)
        ray_pub = self.create_publisher(MarkerArray, 'ray2d', 10)

        loc = self.depth_hypothesis_2d['location']
        depths = self.depth_hypothesis_2d['depths']
        cam_indices = self.depth_hypothesis_2d['camera_indices']
        cams = [self.cameras[idx] for idx in cam_indices]

        # location marker
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = '2d_intersection'
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0)
        marker.pose.position.x = loc[0]
        marker.pose.position.y = loc[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        hypothesis_2d_loc_pub.publish(marker)

        # ray markers
        ray_markers = MarkerArray()
        for i, (cam, depth) in enumerate(zip(cams, depths)):
            ray = cam.box_ray
            start_pt = ray.origin[:2]
            end_pt = ray.origin[:2] + ray.direction[:2] * depth

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f'2d_ray_{i+1}'
            marker.id = i + 3
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.lifetime = Duration(sec=0)
            marker.points = [
                Point(),  # start point
                Point()   # end point
            ]
            marker.points[0].x = start_pt[0]
            marker.points[0].y = start_pt[1]
            marker.points[0].z = 0.0
            marker.points[1].x = end_pt[0]
            marker.points[1].y = end_pt[1]
            marker.points[1].z = 0.0
            marker.scale.x = 0.5
            marker.scale.y = 0.8
            marker.scale.z = 3.0
            marker.color.r = 0.25
            marker.color.g = 0.67
            marker.color.b = 0.73
            marker.color.a = 1.0
            ray_markers.markers.append(marker)

        ray_pub.publish(ray_markers)

        self.get_logger().info("Published 2D ray markers")

    def publish_3d_hypothesis(self):
        """
        Publish a marker for the 3D ray intersection.
        Also publish ray markers for the two cameras used.
        """
        hypothesis_3d_loc_pub = self.create_publisher(Marker, 'hypothesis_3d_location', 10)
        ray_pub = self.create_publisher(MarkerArray, 'ray3d', 10)

        loc = self.depth_hypothesis_3d['location']
        depths = self.depth_hypothesis_3d['depths']
        cam_indices = self.depth_hypothesis_3d['camera_indices']
        cams = [self.cameras[idx] for idx in cam_indices]

        # location marker
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = '3d_intersection'
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0)
        marker.pose.position.x = loc[0]
        marker.pose.position.y = loc[1]
        marker.pose.position.z = loc[2]
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        hypothesis_3d_loc_pub.publish(marker)

        # ray markers
        ray_markers = MarkerArray()
        for i, (cam, depth) in enumerate(zip(cams, depths)):
            ray = cam.box_ray
            start_pt = ray.origin
            end_pt = ray.origin + ray.direction * depth

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f'3d_ray_{i+1}'
            marker.id = i + 3
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.lifetime = Duration(sec=0)
            marker.points = [
                Point(),  # start point
                Point()   # end point
            ]
            marker.points[0].x = start_pt[0]
            marker.points[0].y = start_pt[1]
            marker.points[0].z = start_pt[2]
            marker.points[1].x = end_pt[0]
            marker.points[1].y = end_pt[1]
            marker.points[1].z = end_pt[2]
            marker.scale.x = 0.5
            marker.scale.y = 0.8
            marker.scale.z = 3.0
            marker.color.r = 0.73
            marker.color.g = 0.67
            marker.color.b = 0.25
            marker.color.a = 1.0
            ray_markers.markers.append(marker)

        ray_pub.publish(ray_markers)

        self.get_logger().info("Published 3D ray markers")


def main():
    logger = rclpy.logging.get_logger('logger')
    logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)

    rclpy.init()
    config = OmegaConf.create({})
    node = VizDemo3DTriangulator(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt or SystemExit:
        logger.info("Shutting down...")
        pass

    rclpy.shutdown()
