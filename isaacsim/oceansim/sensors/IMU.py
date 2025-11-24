from isaacsim.sensors.physics import IMUSensor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
import rclpy
import time

class IMU(IMUSensor):
    def __init__(self,
    prim_path,
    name="imu",
    frequency=200, # or, dt=1./60
    translation=None, # or, position=np.array([0, 0, 0]),
    orientation=None,
    linear_acceleration_filter_size = 10,
    angular_velocity_filter_size = 10,
    orientation_filter_size = 10,
    ):
        """Initialize an IMU sensor with configurable parameters.

        .. note::

            This class is inherited from ``IMUSensor``.

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "imu".
            frequency (int, optional): update frequency of the IMU sensor in Hz. Defaults to 60.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            linear_acceleration_filter_size (int, optional): Size of the moving average filter for linear acceleration. Defaults to 10.
            angular_velocity_filter_size (int, optional): Size of the moving average filter for angular velocity. Defaults to 10.
            orientation_filter_size (int, optional): Size of the moving average filter for orientation. Defaults to 10.
        """
        super().__init__(
            prim_path=prim_path,
            name=name,
            frequency=frequency,
            translation=translation,
            orientation=orientation,
            linear_acceleration_filter_size=linear_acceleration_filter_size,
            angular_velocity_filter_size=angular_velocity_filter_size,
            orientation_filter_size=orientation_filter_size
        )

    def get_imu_data(self):
        """
        Retrieve the current IMU data including linear acceleration, angular velocity, and orientation.
        Returns:
            dict: A dictionary containing 'linear_acceleration', 'angular_velocity', and 'orientation'"""
        raw = self.get_current_frame(read_gravity=True)
        imu_data = {
            'linear_acceleration': np.array(raw['lin_acc']),
            'linear_acceleration_covariance': self._linear_acceleration_covariance,
            'angular_velocity':    np.array(raw['ang_vel']),
            'angular_velocity_covariance': self._angular_velocity_covariance,
            'orientation':         np.array(raw['orientation']),
            'orientation_covariance': self._orientation_covariance,
            'time':                raw['time'],
            'physics_step':        raw['physics_step'],
        }

        if self._enable_ros2_pub:
            self._ros2_publish_imu(imu_data)

        return imu_data

    def initialize(self, 
                    enable_ros2_pub=True,
                    imu_topic="/oceansim/robot/imu",
                    ros2_pub_frequency=200,
                    orientation_covariance=np.zeros(9),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration_covariance=np.zeros(9),
                ):

        self._orientation_covariance = orientation_covariance
        self._angular_velocity_covariance = angular_velocity_covariance
        self._linear_acceleration_covariance = linear_acceleration_covariance

        # ROS2 configuration
        self._enable_ros2_pub = enable_ros2_pub
        self._imu_topic = imu_topic
        self._last_publish_time = 0.0
        self._ros2_pub_frequency = ros2_pub_frequency     # publish frequency, hz
        self._setup_ros2_publisher()
        
        print(f'[{self.name}] Initialized successfully.')

    def _setup_ros2_publisher(self):
        '''
        Setup the publisher for IMU
        '''
        try:
            if not self._enable_ros2_pub:
                return
            
            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self.name}] ROS2 context initialized')

            # Create IMU publisher node
            node_name = f'oceansim_rob_imu_pub_{self.name.lower()}'.replace(' ', '_')
            self._ros2_imu_node = rclpy.create_node(node_name)
            self._imu_pub = self._ros2_imu_node.create_publisher(
                Imu, 
                self._imu_topic,
                10
            )
        
        except Exception as e:
            print(f'[{self.name}] ROS2 IMU publisher setup failed: {e}')

    def _ros2_publish_imu(self, imu_data):
        """
        publish the IMU data
        """
        try:
            if self._imu_pub is None:
                return

            # fps control
            current_time = time.time()
            # if current_time - self._last_publish_time < (1.0 / self._frequency):
            #     return

            # Create a ROS2 Imu message
            msg = Imu()
            # msg.header.stamp = self._ros2_imu_node.get_clock().now().to_msg()
            sim_time_sec = imu_data['time']  # This comes from IsaacSim
            msg.header.stamp.sec = int(sim_time_sec)
            msg.header.stamp.nanosec = int((sim_time_sec - int(sim_time_sec)) * 1e9)
            msg.header.frame_id = 'imu'
            orientation = imu_data['orientation']
            msg.orientation = Quaternion(
                x=float(orientation[0]),
                y=float(orientation[1]),
                z=float(orientation[2]),
                w=float(orientation[3])
            )
            msg.orientation_covariance = imu_data['orientation_covariance'].tolist()
            ang_vel = imu_data['angular_velocity']
            msg.angular_velocity = Vector3(
                x=float(ang_vel[0]),
                y=float(ang_vel[1]),
                z=float(ang_vel[2])
            )
            msg.angular_velocity_covariance = imu_data['angular_velocity_covariance'].tolist()
            lin_acc = imu_data['linear_acceleration']
            msg.linear_acceleration = Vector3(
                x=float(lin_acc[0]),
                y=float(lin_acc[1]),
                z=float(lin_acc[2])
            )
            msg.linear_acceleration_covariance = imu_data['linear_acceleration_covariance'].tolist()
            
            # Publish the message
            self._imu_pub.publish(msg)

            rclpy.spin_once(self._ros2_imu_node, timeout_sec=0.0)

            self._last_publish_time = current_time

            # debug
            # self._ros2_uw_imu_node.get_logger().info(
            #     f'Published image: encoding={msg.encoding}, '
            #     f'width={msg.width}, height={msg.height}, step={msg.step}, '
            #     f'data_size={len(msg.data)}'
            # )

        except Exception as e:
            print(f'[{self.name}] ROS2 IMU publish failed: {e}')
            
# usage example:
# imu_sensor = IMU(prim_path="/World/Robot/IMU", name="imu", frequency=100)
# imu_data = imu_sensor.get_imu_data()