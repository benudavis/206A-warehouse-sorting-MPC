from setuptools import find_packages, setup
from glob import glob

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-tah',
    maintainer_email='danielmunicio360@gmail.com',
    description='Planning package for UR7e pick-and-place with IK and MPC',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Main executable (uses MPC for trajectory planning)
            'main_mpc = planning.main_mpc:main',
            'main_mpc_new = planning.main_mpc_new:main',
            'main_waypoints = planning.main_waypoints:main',
            'tf = planning.static_tf_transform:main',
            'ik = planning.ik:main',
            'transform_perception = planning.transform_perception:main',
            'forward_kinematics_node = planning.forward_kinematics_node:main',
            'mpc_visualization = planning.mpc_visualization_node:main',
        ],
    },
)
