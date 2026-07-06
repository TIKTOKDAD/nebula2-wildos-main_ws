"""dlio_odom_twist_adapter 的 ament_python 安装定义."""

from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = "dlio_odom_twist_adapter"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml", "README_CN.md"]),
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
    description="Convert DLIO world-frame odometry twist into base_link.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dlio_odom_twist_adapter = "
            "dlio_odom_twist_adapter.adapter_node:main",
        ],
    },
)
