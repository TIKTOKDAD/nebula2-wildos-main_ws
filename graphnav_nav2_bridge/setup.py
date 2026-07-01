from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = "graphnav_nav2_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="omar",
    maintainer_email="omar@example.com",
    description="Execute graphnav look-ahead goals with Nav2 on the A300.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "goal_pose_to_nav2 = "
            "graphnav_nav2_bridge.goal_pose_to_nav2:main",
        ],
    },
)
