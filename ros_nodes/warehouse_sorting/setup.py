from setuptools import setup
import os
from glob import glob

package_name = 'warehouse_sorting'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Parham Sharaf',
    maintainer_email='parham2002sh@gmail.com',
    description='MPC-based warehouse sorting for UR5e',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller = warehouse_sorting.mpc_controller_node:main',
            'ik_solver = warehouse_sorting.ik_solver_node:main',
            'gripper_controller = warehouse_sorting.gripper_controller_node:main',
            'task_planner = warehouse_sorting.task_planner_node:main',
        ],
    },
)
