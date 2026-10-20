from setuptools import setup

package_name = "seg2grasp_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/seg2grasp.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sunwei-ksdy-robot",
    maintainer_email="sunweijm.sun@gmail.com",
    description="ROS 2 node for Seg2Grasp: live suction-point detection from RealSense / Orbbec topics.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "suction_node = seg2grasp_ros.suction_node:main",
        ],
    },
)
