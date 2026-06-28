from copy import deepcopy
import numpy as np
import cv2
import time
from cv_bridge import CvBridge
from omegaconf import OmegaConf

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from builtin_interfaces.msg import Duration

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster

from .pcl_utils import combine_pointcloud2
from .random_cam_initializer import RandomCameraInitialization
from .bbox_generator import BoundingBoxGenerator
from .particle_generator import ParticleGenerator, ParticleDistribution
from .triangulator import Triangulator
from .camera_data import Camera, Ray


class TriangulatorMetrics(Node):
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
        "particle_generator_config": {
            "num_particles": 1000,  # Number of particles to generate
            "depth_range": [1.0, 200.0],  # Depth range
            "add_odom_drift": True,  # Whether to add odometry drift to the camera pose
            "t_drift_std": 1.0,  # Standard deviation of translation drift
            "r_drift_std": 2.0,  # Standard deviation of rotation drift in degrees
            "hypothesis_std": 100.0,  # Standard deviation from the depth hypothesis for sampling particles
        },
        "triangulator_config": {
            "triangulation_uncertainty": 1.0,
            "triangulation_method": "WEIGHTED"
        },

        "particle_distribution": "UNIFORM_HYP", # "UNIFORM" or "GAUSSIAN_HYP" or "UNIFORM_HYP"
        "num_rays_for_hypothesis": -1,  # Number of rays to use for depth hypothesis (-1 for all)
        "use_2d_hypothesis": True,  # Whether to use a 2D ray intersection hypothesis
        "num_setups": 100,  # Number of setups to average over for triangulation
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
        self.num_rays_for_hypothesis = config.num_rays_for_hypothesis
        if self.num_rays_for_hypothesis == -1:
            self.num_rays_for_hypothesis = self.camera_initialization.num_cameras
        self.use_2d_hypothesis = config.use_2d_hypothesis
        self.num_setups = config.num_setups
        
        self.triangulator = Triangulator(config.triangulator_config)
        self.triangulated_position = np.zeros(3, dtype=np.float64)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

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

        self.errors = []
        self.inference_times = []

    def get_new_setup(self):
        """
        Get a new setup of cameras and triangulate the object location.
        This is useful for resetting the visualization with a new camera configuration.
        """
        self.get_logger().info("Reinitializing cameras and triangulating object location...")
        self.initialize_cameras()

        now = time.time()
        self.triangulated_position = self.triangulator.triangulate(self.cameras)
        inference_time = time.time() - now

        self.get_logger().info(f"Triangulated position: {self.triangulated_position}")
        self.publish_triangulated_marker()
        if self.use_2d_hypothesis:
            self.publish_2d_hypothesis()

        return {
            "error": np.linalg.norm(self.triangulated_position - self.object_loc_world),
            "inference_time": inference_time,
        }

    def publisher(self):
        """
        Publish the data for all the cameras: camera info and image.
        """

        triangulation_data = self.get_new_setup()
        self.errors.append(triangulation_data['error'])
        self.inference_times.append(triangulation_data['inference_time'])
        
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
        if self.use_2d_hypothesis:
            self.publish_2d_hypothesis()

        if len(self.errors) >= self.num_setups:
            avg_error = np.mean(self.errors)
            avg_inference_time = np.mean(self.inference_times)
            self.get_logger().info(f"Average error: {avg_error:.4f} m")
            self.get_logger().info(f"Average inference time: {avg_inference_time:.4f} s")
            std_error = np.std(self.errors)
            std_inference_time = np.std(self.inference_times)
            self.get_logger().info(f"Standard deviation of error: {std_error:.4f} m")
            self.get_logger().info(f"Standard deviation of inference time: {std_inference_time:.4f} s")
            self.errors.clear()
            self.inference_times.clear()

            self.timer.cancel()

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
            self.tf_broadcaster.sendTransform(camera_tf)
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
        
        # Get a depth hypothesis from 2 random cameras
        self.depth_hypothesis = self.get_2d_hypothesis(self.cameras) if self.use_2d_hypothesis else None

        # Generate particles for each camera
        for camera in self.cameras:
            camera = self.particle_generator.generate_particles(
                camera,
                depth_hypothesis=self.depth_hypothesis,
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

    def publish_2d_hypothesis(self):
        """
        Publish a marker for the 2D ray intersection.
        Also publish ray markers for the two cameras used.
        """
        triangulated_2d_loc_pub = self.create_publisher(Marker, 'triangulated_2d_location', 10)
        ray_pub = self.create_publisher(MarkerArray, 'ray2d', 10)

        loc = self.depth_hypothesis['location']
        depths = self.depth_hypothesis['depths']
        cam_indices = self.depth_hypothesis['camera_indices']
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
        triangulated_2d_loc_pub.publish(marker)

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

def main():
    logger = rclpy.logging.get_logger('logger')
    logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)

    rclpy.init()
    config = OmegaConf.create({})
    node = TriangulatorMetrics(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt or SystemExit:
        logger.info("Shutting down...")
        pass

    rclpy.shutdown()
