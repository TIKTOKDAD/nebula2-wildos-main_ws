from setuptools import find_packages, setup

package_name = 'triangulation3d'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hardik Shah',
    maintainer_email='hardikns@jpl.nasa.gov',
    description='Particle filter based 3D triangulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'triangulation_visualizer = triangulation3d.multicam_visualizer:main',
            'teleop_triangulation = triangulation3d.teleop_triangulation:main',
            'teleop_twist_keyboard = triangulation3d.teleop_twist_keyboard:main',
        ],
    },
)
