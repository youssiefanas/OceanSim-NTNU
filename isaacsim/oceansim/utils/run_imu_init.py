import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
import time

class UnderwaterInitPublisher(Node):
    def __init__(self):
        super().__init__('underwater_imu_init_node')
        
        self.publisher_ = self.create_publisher(Twist, '/oceansim/robot/vel_cmd', 10)
        
        # 30Hz update rate
        self.timer_period = 1.0 / 30.0  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.start_time = time.time()
        self.get_logger().info('IMU Initialization Node Started.')
        self.get_logger().info('Phase 0: DESCENDING to find visual features...')

    def timer_callback(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        msg = Twist()

        # --- PHASE 0: DESCENT (0 - 5 seconds) ---
        # "Go down a bit" to find features.
        # Assuming Z-up coordinate system (Standard ROS/Isaac), -Z is down.
        if elapsed < 5.0:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = -0.3  # Dive at 0.3 m/s (~1.5 meters depth)
            msg.angular.z = 0.0

        # --- PHASE 1: STATIC (5 - 15 seconds) ---
        # Station Keeping for 10 seconds.
        # Now that we (hopefully) see features, we stay still to capture biases.
        elif elapsed < 15.0:
            if elapsed < 5.1:
                self.get_logger().info('Phase 1: STATIC (Station Keeping)...')
            
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            
        # --- PHASE 2: DYNAMIC EXCITATION (15 - 65 seconds) ---
        # 50 seconds of active movement
        elif elapsed < 65.0:
            if elapsed < 15.1:
                 self.get_logger().info('Phase 2: DYNAMIC (Excitation Maneuver)...')

            # Reset dynamic timer so t=0 starts at 15s
            t_dynamic = elapsed - 15.0
            
            # --- Tuning Parameters ---
            # Low frequency for smooth, observable motion
            freq_sway = 0.1    
            freq_heave = 0.1   
            freq_yaw = 0.05    
            
            # High amplitude for good signal-to-noise ratio
            amp_sway = 0.8     # m/s 
            amp_heave = 0.4    # m/s 
            amp_yaw = 0.3      # rad/s

            # 1. Sway (Side-to-Side)
            msg.linear.y = amp_sway * math.sin(2 * math.pi * freq_sway * t_dynamic)

            # 2. Heave (Vertical Bobbing)
            msg.linear.z = amp_heave * math.cos(2 * math.pi * freq_heave * t_dynamic)

            # 3. Surge (Forward Crawl)
            # Move forward to explore the map we just found
            msg.linear.x = 0.2 

            # 4. Yaw (Scanning)
            msg.angular.z = amp_yaw * math.sin(2 * math.pi * freq_yaw * t_dynamic)

        # --- FINISH ---
        else:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('Initialization Complete. Stopping.')
            self.destroy_node()
            return

        self.publisher_.publish(msg)

def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    
    node = UnderwaterInitPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
        node.destroy_node()

if __name__ == '__main__':
    main()