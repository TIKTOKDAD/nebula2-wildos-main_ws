import numpy as np
import random
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import matplotlib.pyplot as plt  


FIELDS = [
    PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
]


def float_to_rgb(float_rgb):
    """
    Unpack a float32 RGB value into its constituent R, G, B components.
    """
    rgb_int = struct.unpack('I', struct.pack('f', float_rgb))[0]
    r = (rgb_int >> 16) & 0xFF
    g = (rgb_int >> 8) & 0xFF
    b = rgb_int & 0xFF
    return r, g, b


def rgb_to_float(r, g, b):
    """Pack 3 bytes (RGB) into a single float32 for PointCloud2."""
    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack('f', struct.pack('I', rgb))[0]


def create_colored_pointcloud2(points_3d: np.ndarray, frame_id: str, color: np.ndarray = None) -> PointCloud2:
    header = Header()
    header.frame_id = frame_id

    # Choose a random color if not provided
    if color is None:
        cmap = plt.get_cmap('nipy_spectral')  # or 'Set2', 'Pastel1'
        color = np.array(cmap(random.random())[:3]) * 255  # RGB in 0-255

    r, g, b = color.astype(np.uint8)
    rgb_float = rgb_to_float(r, g, b)

    # Build point cloud data: [x, y, z, rgb]
    pointcloud_data = [
        (pt[0], pt[1], pt[2], rgb_float) for pt in points_3d
    ]

    pcl_msg = point_cloud2.create_cloud(header, FIELDS, pointcloud_data)
    return pcl_msg


def combine_pointcloud2(clouds: list[PointCloud2]) -> PointCloud2:
    """Combine multiple PointCloud2 messages into one."""
    all_points = []

    for cloud in clouds:
        points = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        all_points.extend(points)

    header = Header()
    combined_pcl_msg = point_cloud2.create_cloud(header, FIELDS, all_points)

    return combined_pcl_msg