"""Start the camera (optional) and the Seg2Grasp suction node.

    ros2 launch seg2grasp_ros seg2grasp.launch.py                      # RealSense D435i (default)
    ros2 launch seg2grasp_ros seg2grasp.launch.py camera:=gemini305    # Orbbec Gemini 305
    ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false   # camera already running
    ros2 launch seg2grasp_ros seg2grasp.launch.py roi:=20,140,440,500 vacuum_radius:=20.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Per camera: driver launch file, its arguments, the topics the node subscribes to, and the
# default depth band (mm) of object points used for grasping.
CAMERAS = {
    "d435i": {
        "package": "realsense2_camera",
        "launch_file": "rs_launch.py",
        "launch_arguments": {
            "pointcloud.enable": "true",
            "depth_module.depth_profile": "848x480x30",
            "rgb_camera.color_profile": "848x480x30",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
            "hole_filling_filter.enable": "true",
        },
        "topics": {
            "color_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/depth/image_rect_raw",
            "color_info_topic": "/camera/camera/color/camera_info",
            "depth_info_topic": "/camera/camera/depth/camera_info",
        },
        "z_near": "300.0",
        "z_far": "1050.0",
    },
    # enable_colored_point_cloud makes the driver align depth to color itself (depth_registration),
    # so depth/image_raw is already on the color grid and the node skips its own registration.
    "gemini305": {
        "package": "orbbec_camera",
        "launch_file": "gemini_301_series.launch.py",
        "launch_arguments": {
            "enable_colored_point_cloud": "true",
            "depth_registration": "true",
        },
        "topics": {
            "color_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "color_info_topic": "/camera/color/camera_info",
            "depth_info_topic": "/camera/depth/camera_info",
        },
        "z_near": "100.0",      # short-range camera, typically mounted ~0.3 m above the parts
        "z_far": "1050.0",
    },
}


def launch_setup(context):
    cam = CAMERAS[LaunchConfiguration("camera").perform(context)]
    z_near = LaunchConfiguration("z_near").perform(context) or cam["z_near"]
    z_far = LaunchConfiguration("z_far").perform(context) or cam["z_far"]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(cam["package"]), "launch", cam["launch_file"]])),
        launch_arguments=cam["launch_arguments"].items(),
        condition=IfCondition(LaunchConfiguration("launch_camera")),
    )
    node = Node(
        package="seg2grasp_ros",
        executable="suction_node",
        name="seg2grasp",
        output="screen",
        emulate_tty=True,
        parameters=[{
            **cam["topics"],
            "roi": ParameterValue(LaunchConfiguration("roi"), value_type=str),
            "vacuum_radius": LaunchConfiguration("vacuum_radius"),
            "seal_min": LaunchConfiguration("seal_min"),
            "show": LaunchConfiguration("show"),
            "fp16": LaunchConfiguration("fp16"),
            "seg_min_size": LaunchConfiguration("seg_min_size"),
            "z_near": float(z_near),
            "z_far": float(z_far),
        }],
    )
    return [camera, node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="d435i", choices=list(CAMERAS),
                              description="camera to use: d435i (RealSense D435i) or gemini305 (Orbbec Gemini 305)"),
        DeclareLaunchArgument("launch_camera", default_value="true", description="also start the camera driver"),
        DeclareLaunchArgument("roi", default_value="", description="color-image crop 'y,x,h,w' (empty = full frame)"),
        DeclareLaunchArgument("vacuum_radius", default_value="30.0", description="suction cup radius (mm)"),
        DeclareLaunchArgument("seal_min", default_value="0.98",
                              description="min fraction of the cup disc that must seal (0 = no seal check)"),
        DeclareLaunchArgument("show", default_value="true", description="show the result window"),
        DeclareLaunchArgument("fp16", default_value="true", description="float16 segmentation inference (~3x faster)"),
        DeclareLaunchArgument("seg_min_size", default_value="0",
                              description="segmentation input short edge in px; 0 = model default 800 (smaller = faster, coarser)"),
        DeclareLaunchArgument("z_near", default_value="",
                              description="nearest object depth (mm) used for grasping; empty = camera default "
                                          "(d435i 300, gemini305 100)"),
        DeclareLaunchArgument("z_far", default_value="",
                              description="farthest object depth (mm) used for grasping; empty = camera default (1050)"),
        OpaqueFunction(function=launch_setup),
    ])
