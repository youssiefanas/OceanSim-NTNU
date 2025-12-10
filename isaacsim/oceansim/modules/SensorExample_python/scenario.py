# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, Usd, UsdPhysics
import time
import rclpy

from geometry_msgs.msg import Quaternion, Vector3, Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
import omni.timeline

from isaacsim.asset.gen.omap.bindings import _omap

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw

from isaacsim.oceansim.sensors.datacollection import DataCollectionSensor

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
    def __init__(self, publish_pose=True, publish_map=False):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
        self._data_collection_mode = False
        self.data_collection_path = ""
        self.waypoints_control_speed = False


        self._ctrl_mode = None

        self._running_scenario = False
        self._time = 0.0

        # ROS2 Control
        self._ros2_control_receiver = None
        self._enable_ros2_control = True
        self._ros2_control_mode = "velocity control"

        self._rob_pose_topic = "/oceansim/robot/pose"
        self._publish_pose = publish_pose
        self._publish_map = publish_map

        # Initialize ROS2 context if not already done
        if not rclpy.ok():
            rclpy.init()
            print('ROS2 context initialized')

        if self._publish_pose:
            # Create pose publisher node
            node_name = f'oceansim_rob_pose_pub'
            self._ros2_rob_pose_node = rclpy.create_node(node_name)
            self._rob_pose_pub = self._ros2_rob_pose_node.create_publisher(
                PoseStamped,
                self._rob_pose_topic,
                10
            )

            # Path Publisher
            self._path_topic = "/oceansim/robot/path"
            self._path_pub = self._ros2_rob_pose_node.create_publisher(
                Path,
                self._path_topic,
                10
            )
            self._path_msg = Path()
            self._path_msg.header.frame_id = 'map' # Path is in map frame

            self._last_path_pos = None
            self._path_update_threshold = 0.05  # Only add point if moved self._path_update_threshold meters

            # TF Broadcaster
            self._tf_broadcaster = TransformBroadcaster(self._ros2_rob_pose_node)

            if self._publish_map:
                self._omap_generator = None
                self._last_omap_pos = None
                self._omap_update_threshold = 0.5
                
                # Initialize Debug Draw interface (Safe to do here)
                if _debug_draw is not None:
                    self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                else:
                    self._debug_draw = None

        self._timeline = omni.timeline.get_timeline_interface()

    def setup_scenario(self, rob, sonar, cam, DVL, baro, IMU, ctrl_mode,data_collection_mode, data_collection_path=""):
        if not rclpy.ok():
            print("[Scenario] ROS2 Context was dead. Resurrecting before sensor init...")
            rclpy.init()

        self._data_collection_mode = data_collection_mode
        self.data_collection_path = data_collection_path
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._IMU = IMU
        self._IMU = IMU
        self._ctrl_mode = ctrl_mode
        
        # Initialize ROS2 publishers for DVL and Barometer
        if self._DVL is not None:
            self._DVL.initialize_ros2()
        
        if self._baro is not None:
            self._baro.initialize_ros2()

        # Data collection setup
        if self._data_collection_mode:
            self.setup_data_collection(data_path=self.data_collection_path)
            if self._sonar is not None:
                sensor_name = "sonar_sensor"
                sensor_path = self._data_collector.collect_data(name=sensor_name)
                self._sonar.sonar_initialize(include_unlabelled=True)
            if self._cam is not None:
                sensor_name = "camera_sensor"
                sensor_path = self._data_collector.collect_data(name=sensor_name)
                self._cam.initialize(writing_dir=sensor_path)#UW_yaml_path='/home/osim-mir/OceanSimAssets/water_params_test.yaml')#, )
            if self._DVL is not None:
                sensor_name = "DVL_sensor"
                sensor_path = self._data_collector.collect_data(name=sensor_name)
                self._DVL_reading = [0.0, 0.0, 0.0]
            if self._baro is not None:
                sensor_name = "barometer_sensor"
                sensor_path = self._data_collector.collect_data(name=sensor_name)
                self._baro_reading = 101325.0 # atmospheric pressure (Pa)
            if self._IMU is not None:
                sensor_name = "IMU_sensor"
                sensor_path = self._data_collector.collect_data(name=sensor_name)
                self._IMU.initialize()
                self._IMU.save_metadata(save_path=sensor_path)
                self._IMU.init_logging(save_path=sensor_path) # <--- Init CSV Logging
                self._IMU_reading = {
                    'linear_acceleration': np.array([0.0, 0.0, 0.0]),
                    'angular_velocity':    np.array([0.0, 0.0, 0.0]),
                    'orientation':         np.array([1.0, 0.0, 0.0, 0.0]),
                    'time':                0.0,
                    'physics_step':        0    
                }
        else:
            if self._sonar is not None:
                self._sonar.sonar_initialize(include_unlabelled=True)
            if self._cam is not None:
                self._cam.initialize()#UW_yaml_path='/home/osim-mir/OceanSimAssets/water_params_test.yaml')#, writing_dir="/home/osim-mir/OceanSimAssets/GroundTruth")
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

        try:
            self._physx_interface = omni.physx.acquire_physx_interface()
            self._stage_id = omni.usd.get_context().get_stage_id()
            
            # Check if PhysicsScene exists (Optional safety check)
            stage = omni.usd.get_context().get_stage()
            has_physics_scene = False
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    has_physics_scene = True
                    break
            
            if has_physics_scene:
                self._omap_generator = _omap.Generator(self._physx_interface, self._stage_id)
                self._omap_generator.update_settings(0.2, 4, 5, 6)
                print("[Scenario] Occupancy Map Generator Initialized.")
            else:
                print("[Scenario] WARNING: No PhysicsScene found on stage. OMap disabled.")
        except Exception as e:
            print(f"[Scenario] Failed to init OMap Generator: {e}")
            self._omap_generator = None

        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control" or ctrl_mode == "ROS + Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd
            from ...utils.gamepad_cmd import gamepad_cmd
            import carb.input
            from ...utils.gamepad_cmd import gamepad_cmd

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
            self._joy_force = gamepad_cmd(
                input_mapping={
                    carb.input.GamepadInput.LEFT_STICK_UP:    np.array([1.0, 0.0, 0.0]),   # Forward
                    carb.input.GamepadInput.LEFT_STICK_DOWN:  np.array([-1.0, 0.0, 0.0]),  # Backward
                    carb.input.GamepadInput.LEFT_STICK_LEFT:  np.array([0.0, 1.0, 0.0]),   # Left
                    carb.input.GamepadInput.LEFT_STICK_RIGHT: np.array([0.0, -1.0, 0.0]),  # Right
                    carb.input.GamepadInput.RIGHT_TRIGGER:    np.array([0.0, 0.0, 1.0]),   # Up
                    carb.input.GamepadInput.LEFT_TRIGGER:     np.array([0.0, 0.0, -1.0]),  # Down
                },
                scale=9.0 
            )
            
            self._joy_torque = gamepad_cmd(
                input_mapping={
                    # Pitch
                    carb.input.GamepadInput.RIGHT_STICK_UP:    np.array([0.0, -1.0, 0.0]), 
                    carb.input.GamepadInput.RIGHT_STICK_DOWN:  np.array([0.0, 1.0, 0.0]),
                    # Yaw
                    carb.input.GamepadInput.RIGHT_STICK_LEFT:  np.array([0.0, 0.0, 1.0]),
                    carb.input.GamepadInput.RIGHT_STICK_RIGHT: np.array([0.0, 0.0, -1.0]),
                    # Roll
                    carb.input.GamepadInput.LEFT_SHOULDER:     np.array([-1.0, 0.0, 0.0]),
                    carb.input.GamepadInput.RIGHT_SHOULDER:    np.array([1.0, 0.0, 0.0]),
                },
                scale=4.0
            )

        if ctrl_mode == "ROS control" or ctrl_mode == "ROS + Manual control":
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

    def setup_data_collection(self, data_path):
        if data_path is None or data_path=="":
            data_path = "/home/osim-mir/data_collected_oceansim"
        else:
            data_path = data_path
        self.data_collection_path = data_path
        try:
            self._data_collector = DataCollectionSensor(data_path=self.data_collection_path)
        except Exception as e:
            print(f"[Scenario] Error initializing DataCollectionSensor: {e}")
            import traceback
            traceback.print_exc()

        
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()
        if self._IMU is not None:
            self._IMU.close()
        
        if self._DVL is not None:
             self._DVL.cleanup()
             
        if self._baro is not None:
             self._baro.cleanup()
        
        # Reset simple variables
        self._time = 0.0

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control" or self._ctrl_mode=="ROS + Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()
            # self._joy_force.cleanup()
            # self._joy_torque.cleanup()

        # clear the ROS2 control receiver
        if self._ros2_control_receiver is not None:
            self._ros2_control_receiver.close()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
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

            # Publish TF (Transform from 'map' to 'base_link')
            tf_msg = TransformStamped()
            tf_msg.header.stamp = msg.header.stamp
            tf_msg.header.frame_id = 'map'
            tf_msg.child_frame_id = 'base_link' # Assuming robot frame is base_link

            tf_msg.transform.translation.x = msg.pose.position.x
            tf_msg.transform.translation.y = msg.pose.position.y
            tf_msg.transform.translation.z = msg.pose.position.z
            tf_msg.transform.rotation = msg.pose.orientation

            self._tf_broadcaster.sendTransform(tf_msg)

            # Publish Path (Full trajectory)
            current_pos_np = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            update_path = False
            if self._last_path_pos is None:
                update_path = True
            else:
                # Calculate distance moved
                dist = np.linalg.norm(current_pos_np - self._last_path_pos)
                if dist > self._path_update_threshold:
                    update_path = True
            
            if update_path:
                self._path_msg.header.stamp = msg.header.stamp
                self._path_msg.poses.append(msg)
                self._path_pub.publish(self._path_msg)
                
                # Update tracker
                self._last_path_pos = current_pos_np

            if self._publish_map:
                if self._debug_draw is not None and self._omap_generator is not None:
                    update_omap = False
                    
                    # Check if this is the first run
                    if self._last_omap_pos is None:
                        update_omap = True
                    else:
                        # Check distance from last generation point
                        dist_omap = np.linalg.norm(current_pos_np - self._last_omap_pos)
                        if dist_omap > self._omap_update_threshold:
                            update_omap = True
                    
                    if update_omap:
                        # 1. Update Transform (Center on Robot)
                        self._omap_generator.set_transform(
                            (float(trans[0]), float(trans[1]), float(trans[2])), 
                            (-2.0, -2.0, -2.0), 
                            (2.0, 2.0, 2.0)
                        )
                        
                        # Generate
                        self._omap_generator.generate3d()
                        
                        # Get Points (These are in World Frame)
                        raw_points = self._omap_generator.get_occupied_positions()
                        print(raw_points)
                        
                        if len(raw_points) > 0:
                            points_np = np.array(raw_points)
                            
                            # FILTER: Disregard points over the robot
                            dist_to_robot = np.linalg.norm(points_np - current_pos_np, axis=1)
                            mask = dist_to_robot > 0.6
                            filtered_points = points_np[mask]

                            # Draw
                            if len(filtered_points) > 0:
                                points_list = [tuple(p) for p in filtered_points]
                                colors = [(1, 0, 0, 1)] * len(points_list) # Red
                                sizes = [10.0] * len(points_list)
                                
                                self._debug_draw.draw_points(points_list, colors, sizes)
                        
                        # Update the last position tracker
                        self._last_omap_pos = current_pos_np

        # IMU UPDATE (Fast - 200 Hz)
        # We ALWAYS update the IMU every physics step
        if self._IMU is not None:
            self._IMU_reading = self._IMU.get_imu_data()
            
            # Log data if in collection mode
            if self._data_collection_mode:
                self._IMU.log_data(
                    timestamp=self._time,
                    accel=self._IMU_reading['linear_acceleration'],
                    gyro=self._IMU_reading['angular_velocity']
                )

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
            new_dvl_reading = self._DVL.get_linear_vel_fd(step)
            # Only update if we have a valid reading (not NaN)
            if not np.any(np.isnan(new_dvl_reading)):
                 self._DVL_reading = new_dvl_reading
                 self._DVL.publish_ros2(self._time, self._DVL_reading)

        # BARO UPDATE (Fast - 200 Hz)
        if self._baro is not None:
            self._baro_reading = self._baro.get_pressure()
            # Publish to ROS2
            self._baro.publish_ros2(self._time, self._baro_reading)

        if self._ctrl_mode=="Manual control" or self._ctrl_mode=="ROS + Manual control":
            # Get Keyboard inputs
            kb_force = self._force_cmd._base_command
            kb_torque = self._torque_cmd._base_command
            
            # Get Joystick inputs
            joy_force = self._joy_force._base_command
            joy_torque = self._joy_torque._base_command

            # Combine them (Summing them allows using both simultaneously)
            total_force = kb_force + joy_force
            total_torque = kb_torque + joy_torque

            user_is_controlling = np.linalg.norm(total_force) > 0.001 or np.linalg.norm(total_torque) > 0.001

            if self._ctrl_mode=="ROS + Manual control" and self._ros2_control_receiver is not None and not user_is_controlling:
                    self._ros2_control_receiver.update_control()

            if user_is_controlling:
                force_cmd = Gf.Vec3f(*total_force)
                torque_cmd = Gf.Vec3f(*total_torque)
                self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
                self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
            else:
                self._rob_forceAPI.CreateForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                self._rob_forceAPI.CreateTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                if self._ctrl_mode == "ROS + Manual control" and self._ros2_control_receiver is not None:
                     self._ros2_control_receiver.update_control()
        elif self._ctrl_mode=="Waypoints":
            if self.waypoints_control_speed:
                SPEED = 0.01  # How much to move per frame (0.0 to 1.0)
                ROT_SPEED = 0.01
                THRESHOLD = 0.1 # Distance units to consider "arrived"
                if len(self.waypoints) > 0:
                    target_data = self.waypoints[0]
                    target_pos = Gf.Vec3d(target_data[0], target_data[1], target_data[2])
                    target_rot = Gf.Quatd(target_data[3], target_data[4], target_data[5], target_data[6])

                    current_pos_attr = self._rob.GetAttribute('xformOp:translate')
                    current_rot_attr = self._rob.GetAttribute('xformOp:orient')
                    
                    current_pos = current_pos_attr.Get()
                    current_rot = current_rot_attr.Get()

                    new_pos = current_pos + (target_pos - current_pos) * SPEED
                    
                    new_rot = Gf.Slerp(ROT_SPEED, current_rot, target_rot)

                    current_pos_attr.Set(new_pos)
                    current_rot_attr.Set(new_rot)
                    
                    distance_vector = target_pos - current_pos
                    distance = distance_vector.GetLength()
                    if distance < THRESHOLD:
                        self.waypoints.pop(0)
                else:
                    print('Waypoints finished')  
            else:
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




        

        


