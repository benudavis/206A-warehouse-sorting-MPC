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
            # MPC-first executables (default "main" now points to MPC implementation)
            'main = planning.mpc_pick_place:main',
            'tf = planning.static_tf_transform:main',
            'ik = planning.ik:main',
            'transform_cube_pose = planning.transform_cube_pose:main',

            # Explicit names for MPC and legacy MoveIt-only flows
            'lab7_mpc_pick_place = planning.mpc_pick_place:main',
            'lab7_moveit_pick_place = planning.main:main',
        ],
    },
)
