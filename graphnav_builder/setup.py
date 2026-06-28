"""graphnav_builder 的 ament_python 打包定义。

该文件将运行时代码、launch 文件、默认配置和资源索引一并安装，使用户能够通过
``ros2 run`` 与 ``ros2 launch`` 在安装空间中发现本包。
"""

from glob import glob
import os

from setuptools import find_packages, setup

# 统一包名，避免 data_files、入口点和 setuptools 元数据出现不一致。
package_name = 'graphnav_builder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 索引文件使 ROS 2 能按包名定位 share 目录。
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
        # 安装 launch/config 后，launch 文件可通过 FindPackageShare 找到它们。
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Patrick Spieler',
    maintainer_email='patrick.spieler@jpl.nasa.gov',
    description=(
        'Sparse navigation graph construction for WildOS-style '
        'graph navigation.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        # ``ros2 run graphnav_builder graphnav_builder`` 调用该入口。
        'console_scripts': [
            'graphnav_builder = graphnav_builder.builder_node:main',
        ],
    },
)
