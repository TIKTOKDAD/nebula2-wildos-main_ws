from copy import deepcopy
import numpy as np
import cv2
from cv_bridge import CvBridge
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from builtin_interfaces.msg import Duration

import rclpy
from rclpy.node import Node

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .pcl_utils import combine_pointcloud2
from .random_cam_initializer import RandomCameraInitialization
from .bbox_generator import BoundingBoxGenerator
from .particle_generator import ParticleGenerator
from .triangulator import Triangulator
from .camera_data import Camera, CameraPose, get_extrinsics_from_tfmsg


class TeleopTriangulator(Node):
    """
    Initialize a single camera and teleoperate it to triangulate the object location.
    This node publishes the camera info, image, and point cloud data for the triangulated object.
    It also publishes markers for the object location and the triangulated position.
    """
    default_config = {
        "object_loc_world": [0.0, 0.0, 0.0],  # Location of the object in the world frame
        "object_size": 5.0,  # Size of the object in meters (assumed to be a cube)
        "move_step": 0.1,  # Step size for moving the camera in meters
        "rotate_step": 1.0,  # Step size for rotating the camera in degrees
        "max_cameras": 15,  # Maximum number of cameras to initialize
        "camera_initialization_config": {
            "num_cameras": 1,
            "cam_distance_range": [10.0,30.0],  # Distance range from the object in meters
        },
        "bounding_box_config": {
            "noise_mean": 0.0,
            "noise_std": 2.0,  # Standard deviation of the noise added to the bounding box
        },
        "particle_generator_config": {},
    }

    def __init__(self, config: OmegaConf):
        super().__init__('demo_triangulator')
        config = OmegaConf.merge(OmegaConf.create(self.default_config), config)

        # Initialize node parameters
        self.object_loc_world = np.array(config.object_loc_world, dtype=np.float64)
        self.object_size = config.object_size
        self.move_step = config.move_step
        self.rotate_step = np.deg2rad(config.rotate_step)  # Convert to radians
        self.max_cameras = config.max_cameras

        # External dependencies
        self.camera_initialization = RandomCameraInitialization(self.object_loc_world, config.camera_initialization_config)
        self.bounding_box_generator = BoundingBoxGenerator(self.object_loc_world, self.object_size, config.bounding_box_config)
        self.particle_generator = ParticleGenerator(config.particle_generator_config)
        self.triangulator = Triangulator()

        # ROS2 messaging setup
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.bridge = CvBridge()

        # Initialize the camera
        gen_cam_pose: CameraPose = self.camera_initialization.generate_camera(
            self.object_loc_world, True
        )

        self.cameras = []
        self.cameras.append(self.initialize_camera_from_tf(
            RandomCameraInitialization.campose_to_transform(
                gen_cam_pose, world_frame_id='world', child_frame_id='camera_0'
            )
        ))

        self.current_cam_tf = RandomCameraInitialization.campose_to_transform(
            gen_cam_pose, world_frame_id='world', child_frame_id='current_camera'
        )

        self.get_logger().info(f"Initialized camera_{len(self.cameras)} and published static transforms!")
        self.publish_obj_marker()

        # Trajectory and triangulated position
        self.triangulated_position = None  # Initialize triangulated position to None
        self.path = Path()
        self.path.header.frame_id = 'world'
        
        # publishers and subscribers
        self.twist_subscriber = self.create_subscription(
            Twist, 'cmd_vel', self.move_camera, 10
        )

        self.camera_info_publishers = [
            self.create_publisher(
                CameraInfo,
                f'camera_{idx}/camera_info',
                10
            ) for idx in range(self.max_cameras)
        ]
        self.current_cam_publisher = self.create_publisher(
            CameraInfo,
            'current_camera/camera_info',
            10
        )
        self.image_publishers = [
            self.create_publisher(
                Image,
                f'camera_{idx}/image_raw',
                10
            ) for idx in range(self.max_cameras)
        ]
        self.current_cam_image_publisher = self.create_publisher(
            Image,
            'current_camera/image_raw',
            10
        )

        # visualization publishers
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            'particles',
            10
        )
        self.path_publisher = self.create_publisher(
            Path,
            'camera_path',
            10
        )

        self.timer = self.create_timer(0.1, self.publisher)

    def cam_point_to_tf(self, cam_point: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray):
        cam_point*=self.move_step  # Scale the camera point by the move step
        cam_point_world = R_wc @ cam_point + t_wc
        self.current_cam_tf.transform.translation.x = cam_point_world[0]
        self.current_cam_tf.transform.translation.y = cam_point_world[1]
        self.current_cam_tf.transform.translation.z = cam_point_world[2]

    def rot_to_tf(self, R_wc: np.ndarray, angular: np.ndarray):
        """
        Rotate the camera transform by a given angle around a specified axis in the camera frame.
        The axis can be 'x', 'y', or 'z'.
        """
        if angular[0] != 0:
            axis = 'x'
            angle = angular[0] * self.rotate_step
        elif angular[1] != 0:
            axis = 'y'
            angle = angular[1] * self.rotate_step
        elif angular[2] != 0:
            axis = 'z'
            angle = angular[2] * self.rotate_step
        else:
            # self.get_logger().warn("No rotation specified, skipping rotation.")
            return

        # Create a rotation matrix for the specified axis
        if axis == 'x':
            rot_matrix = R.from_euler('x', angle).as_matrix()
        elif axis == 'y':
            rot_matrix = R.from_euler('y', angle).as_matrix()
        else:
            rot_matrix = R.from_euler('z', angle).as_matrix()

        # rot_matrix is in the camera frame, so we need to apply it to the current rotation
        R_wc_new = rot_matrix @ R_wc
        R_wc_quat = R.from_matrix(R_wc_new).as_quat()

        self.current_cam_tf.transform.rotation.x = R_wc_quat[0]
        self.current_cam_tf.transform.rotation.y = R_wc_quat[1]
        self.current_cam_tf.transform.rotation.z = R_wc_quat[2]
        self.current_cam_tf.transform.rotation.w = R_wc_quat[3]


    def move_camera(self, msg: Twist):
        """
        Move the camera based on the input message.
        """
        linear = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64)
        angular = np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64)
        
        # Get the current camera transform translation and rotation
        cam_extrinsics = get_extrinsics_from_tfmsg(self.current_cam_tf)
        R_wc = cam_extrinsics.R
        t_wc = cam_extrinsics.t

        if linear[0] == 100:    # Special command to initialize a new camera
            if len(self.cameras) < self.max_cameras:
                # Initialize the camera at the current position
                cam_tf = deepcopy(self.current_cam_tf)
                cam_tf.child_frame_id = f'camera_{len(self.cameras)}'
                cam = self.initialize_camera_from_tf(cam_tf)
                self.cameras.append(cam)
                self.get_logger().info(f"Initialized camera_{len(self.cameras)} and published static transforms!")

                # triangulate the object location from the projected particles
                if len(self.cameras) > 1:
                    self.triangulated_position = self.triangulator.triangulate(self.cameras)
                    self.get_logger().info(f"Triangulated position: {self.triangulated_position}")
                    self.publish_triangulated_marker()

            else:
                self.get_logger().warn("Maximum number of cameras reached, cannot initialize more cameras.")
        else:
            self.cam_point_to_tf(linear, R_wc, t_wc)
            self.rot_to_tf(R_wc, angular)

            # update the path
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'world'
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.position.x = self.current_cam_tf.transform.translation.x
            pose_msg.pose.position.y = self.current_cam_tf.transform.translation.y
            pose_msg.pose.position.z = self.current_cam_tf.transform.translation.z
            pose_msg.pose.orientation.x = self.current_cam_tf.transform.rotation.x
            pose_msg.pose.orientation.y = self.current_cam_tf.transform.rotation.y
            pose_msg.pose.orientation.z = self.current_cam_tf.transform.rotation.z
            pose_msg.pose.orientation.w = self.current_cam_tf.transform.rotation.w

            self.path.poses.append(pose_msg)
            self.path.header.stamp = self.get_clock().now().to_msg()
            self.path_publisher.publish(self.path)
            

    def initialize_camera_from_tf(self, camera_tf: TransformStamped) -> Camera:
        """
        Initialize the camera and publish the transforms.
        """
        
        pub_time = self.get_clock().now().to_msg()

        camera_tf.header.stamp = pub_time
        self.tf_static_broadcaster.sendTransform(camera_tf)
        # self.get_logger().info(f"Published transform for {camera_tf.child_frame_id}")

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

        # Generate particles
        camera = self.particle_generator.generate_particles(camera)
        return camera

    def publisher(self):
        """
        Publish the object marker, camera info, image, and point cloud data.
        If the object has been triangulated, publish the triangulated position as a marker.
        """

        # Publish the camera info
        pub_time = self.get_clock().now().to_msg()
        current_cam = self.initialize_camera_from_tf(self.current_cam_tf)

        # publish
        for idx, camera in enumerate(self.cameras):
            camera_info = camera.camera_info
            camera_info.header.stamp = pub_time
            self.camera_info_publishers[idx].publish(camera_info)

            image_msg = camera.image
            image_msg.header.stamp = pub_time
            self.image_publishers[idx].publish(image_msg)

            # self.get_logger().info(f"Published camera info and image for {camera.camera_tf.child_frame_id}")
        # Publish the current camera info and image
        current_cam_info = current_cam.camera_info
        current_cam_info.header.stamp = pub_time
        self.current_cam_publisher.publish(current_cam_info)
        current_cam_image = current_cam.image
        current_cam_image.header.stamp = pub_time
        self.current_cam_image_publisher.publish(current_cam_image)

        # Combine the points from all the cameras into a single PointCloud message
        combined_pcl_msg = self.triangulator.combine_points(self.cameras + [current_cam])
        combined_pcl_msg.header.stamp = pub_time
        self.point_cloud_publisher.publish(combined_pcl_msg)
        # self.get_logger().info("Published combined point cloud message")

        # Publish the object marker
        self.publish_obj_marker()

        # triangulate the object location from the projected particles
        if self.triangulated_position is not None:
            self.publish_triangulated_marker()
        else:
            # self.get_logger().info("Not enough cameras to triangulate the object location.")
            pass

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
        # self.get_logger().info("Published object location")

    def publish_triangulated_marker(self):
        """
        Publish a marker for the triangulated position.
        """
        marker = self.triangulator.get_triangulated_marker(self.triangulated_position)
        marker.header.stamp = self.get_clock().now().to_msg()

        # Publish the triangulated position
        triangulated_pub = self.create_publisher(Marker, 'triangulated_marker', 10)
        triangulated_pub.publish(marker)



def main():
    logger = rclpy.logging.get_logger('logger')
    logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)

    rclpy.init()
    config = OmegaConf.create({})
    node = TeleopTriangulator(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt or SystemExit:
        logger.info("Shutting down...")
        pass

    rclpy.shutdown()
