"""Start the RealSense D435i (optional) and the Seg2Grasp suction node.

    ros2 launch seg2grasp_ros seg2grasp.launch.py
    ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false   # camera already running
    ros2 launch seg2grasp_ros seg2grasp.launch.py roi:=20,140,440,500 vacuum_radius:=20.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])),
        launch_arguments={
            "pointcloud.enable": "true",
            "depth_module.depth_profile": "848x480x30",
            "rgb_camera.color_profile": "848x480x30",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        }.items(),
        condition=IfCondition(LaunchConfiguration("launch_camera")),
    )
    node = Node(
        package="seg2grasp_ros",
        executable="suction_node",
        name="seg2grasp",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "roi": ParameterValue(LaunchConfiguration("roi"), value_type=str),
            "vacuum_radius": LaunchConfiguration("vacuum_radius"),
            "seal_min": LaunchConfiguration("seal_min"),
            "show": LaunchConfiguration("show"),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument("launch_camera", default_value="true", description="also start realsense2_camera"),
        DeclareLaunchArgument("roi", default_value="", description="color-image crop 'y,x,h,w' (empty = full frame)"),
        DeclareLaunchArgument("vacuum_radius", default_value="30.0", description="suction cup radius (mm)"),
        DeclareLaunchArgument("seal_min", default_value="0.98",
                              description="min fraction of the cup disc that must seal (0 = no seal check)"),
        DeclareLaunchArgument("show", default_value="true", description="show the result window"),
        camera,
        node,
    ])
