from isaacsim.sensors.physics import IMUSensor
import numpy as np
import rclpy

class IMU(IMUSensor):
    def __init__(self,
    prim_path,
    name="imu",
    frequency=60, # or, dt=1./60
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
            'angular_velocity':    np.array(raw['ang_vel']),
            'orientation':         np.array(raw['orientation']),
            'time':                raw['time'],
            'physics_step':        raw['physics_step']
        }
        return imu_data
            
# usage example:
# imu_sensor = IMU(prim_path="/World/Robot/IMU", name="imu", frequency=100)
# imu_data = imu_sensor.get_imu_data()