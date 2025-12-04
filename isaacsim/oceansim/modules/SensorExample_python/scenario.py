# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema
import time
import rclpy

from geometry_msgs.msg import Quaternion, Vector3, Pose, PoseStamped

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
import omni.timeline

# ROS Control import
try:
    from isaacsim.oceansim.utils.ros2_control import ROS2ControlReceiver
    ROS2_CONTROL_AVAILABLE = True
    print("[Scenario] Simple ROS2 Control receiver found")
except ImportError as e:
    ROS2_CONTROL_AVAILABLE = False
    print(f"[Scenario] Simple ROS2 Control not available: {e}")
    print("[Scenario] ROS2 Control functionality will be disabled")

class MHL_Sensor_Example_Scenario():
    def __init__(self, publish_pose=True):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None

        self._ctrl_mode = None

        self._running_scenario = False
        self._time = 0.0

        # ROS2 Control
        self._ros2_control_receiver = None
        self._enable_ros2_control = True
        self._ros2_control_mode = "velocity control"

        self._rob_pose_topic = "/oceansim/robot/pose"
        self._publish_pose = publish_pose

        # Initialize ROS2 context if not already done
        if not rclpy.ok():
            rclpy.init()
            print('[ROS2 context initialized')

        if self._publish_pose:
            # Create pose publisher node
            node_name = f'oceansim_rob_pose_pub'
            self._ros2_rob_pose_node = rclpy.create_node(node_name)
            self._rob_pose_pub = self._ros2_rob_pose_node.create_publisher(
                PoseStamped,
                self._rob_pose_topic,
                10
            )

        self._timeline = omni.timeline.get_timeline_interface()

    def setup_scenario(self, rob, sonar, cam, DVL, baro, IMU, ctrl_mode):
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._IMU = IMU
        self._ctrl_mode = ctrl_mode
        if self._sonar is not None:
            self._sonar.sonar_initialize(include_unlabelled=True)
        if self._cam is not None:
            self._cam.initialize()
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)
        if self._IMU is not None:
            self._IMU.initialize()
            self._IMU_reading = {
                'linear_acceleration': np.array([0.0, 0.0, 0.0]),
                'angular_velocity':    np.array([0.0, 0.0, 0.0]),
                'orientation':         np.array([1.0, 0.0, 0.0, 0.0]),
                'time':                0.0,
                'physics_step':        0    
            }        
        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, 10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, 10.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, -10.0],
                                        # pitch command (up)
                                        "I": [0.0, -10.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, 10.0, 0.0],
                                        # row command (left)
                                        "LEFT": [-10.0, 0.0, 0.0],
                                        # row command (negative)
                                        "RIGHT": [10.0, 0.0, 0.0],
                                      })
        elif ctrl_mode == "ROS control":
            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)

            # initialize ROS2ControlReceiver
            self._setup_ros2_control()
            
        self._running_scenario = True

    def _setup_ros2_control(self):
        """setup ROS2 control receiver"""
        if not ROS2_CONTROL_AVAILABLE:
            return
        
        try:
            self._ros2_control_receiver = ROS2ControlReceiver(self._rob, "ROS2ControlReceiver")
            
            if hasattr(self, '_rob_forceAPI') and self._rob_forceAPI is not None:
                self._ros2_control_receiver.set_scenario_force_api(self._rob_forceAPI)

            self._ros2_control_receiver.initialize(
                enable_ros2=True
            )

            self._ros2_control_receiver._setup_ros2_control_mode(
                self._ros2_control_mode
            )
                
        except Exception as e:
            print(f"[Scenario] setup ros2 control receiver failed: {e}")
            self._ros2_control_receiver = None

    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

        
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        # clear the ROS2 control receiver
        if self._ros2_control_receiver is not None:
            self._ros2_control_receiver.close()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step

        # Debug: Check actual update rate
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = time.time()
            self._update_count = 0
        
        self._update_count += 1
        if self._update_count % 100 == 0:  # Print every 100 updates
            current_time = time.time()
            actual_hz = 100 / (current_time - self._last_update_time)
            print(f"Physics callback rate: {actual_hz:.1f} Hz")
            self._last_update_time = current_time

        if self._rob_pose_pub is None:
            return

        # fps control
        # current_time = time.time()
        # if current_time - self._last_publish_time < (1.0 / self._frequency):
        #     return

        if self._publish_pose:
            # Create a ROS2 Imu message
            msg = PoseStamped()

            sim_time = self._timeline.get_current_time()  # Simulation time
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
            msg.header.frame_id = 'map'

            # Get the full 4x4 World Transform Matrix
            world_transform = omni.usd.get_world_transform_matrix(self._rob)

            # Extract Rotation (Quaternion) and Translation
            # method .ExtractRotation() returns a Gf.Rotation, which we convert to Quat
            rot = world_transform.ExtractRotationQuat() 
            trans = world_transform.ExtractTranslation()

            # Populate Message
            msg.pose.position.x = float(trans[0])
            msg.pose.position.y = float(trans[1])
            msg.pose.position.z = float(trans[2])

            msg.pose.orientation.w = float(rot.GetReal())
            msg.pose.orientation.x = float(rot.GetImaginary()[0])
            msg.pose.orientation.y = float(rot.GetImaginary()[1])
            msg.pose.orientation.z = float(rot.GetImaginary()[2])
            
            # Publish the message
            self._rob_pose_pub.publish(msg)

        # IMU UPDATE (Fast - 200 Hz)
        # We ALWAYS update the IMU every physics step
        if self._IMU is not None:
            self._IMU_reading = self._IMU.get_imu_data()

        # CAMERA UPDATE (Slow - 20 Hz)
        # We only render if enough SIMULATION time has passed.
        # Initialize tracker if it doesn't exist
        if not hasattr(self, '_last_cam_time'):
            self._last_cam_time = 0.0
        
        TARGET_CAM_FPS = 20.0
        
        # Check if 0.05s (sim time) has passed since last render
        if (self._time - self._last_cam_time) >= (1.0 / TARGET_CAM_FPS):
            if self._cam is not None:
                self._cam.render(sim_time=self._time)
            
            # Update tracker
            self._last_cam_time = self._time
        
        if self._sonar is not None:
            self._sonar.make_sonar_data()
        if self._DVL is not None:
            self._DVL_reading = self._DVL.get_linear_vel()
        if self._baro is not None:
            self._baro_reading = self._baro.get_pressure()

        if self._ctrl_mode=="Manual control":
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
            self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0])) 
        elif self._ctrl_mode=="ROS control":
            if self._ros2_control_receiver is not None:
                self._ros2_control_receiver.update_control()
            else:
                print("[Scenario] ROS2 Control receiver is not initialized, skipping update.")




        

        


