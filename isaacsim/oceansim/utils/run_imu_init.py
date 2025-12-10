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
        self.get_logger().info('ORB-SLAM3 Initialization Node Started.')
        self.get_logger().info('Strategy: Slower, smoother excitation.')

    def timer_callback(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        msg = Twist()

        # --- PHASE 0: DESCENT (0 - 3 seconds) ---
        # Increased time slightly to allow gentle settling
        if elapsed < 3.0:
            if elapsed < 0.1: self.get_logger().info('Phase 0: DESCENDING...')
            msg.linear.z = -0.2  # Slower descent (was -0.3)
            msg.linear.x = 0.05 

        # --- PHASE 1: GENTLE WAKEUP (3 - 8 seconds) ---
        # Very slow lateral sway to seed the map
        elif elapsed < 8.0:
            if elapsed < 3.1: self.get_logger().info('Phase 1: MAP SEEDING (Slow Lateral Move)...')
            
            # Period = 10 seconds (Very slow)
            msg.linear.y = 0.3 * math.sin(2 * math.pi * 0.1 * (elapsed - 3.0))
            msg.linear.x = 0.1

        # --- PHASE 2: THE "SLOW DANCE" (8 - 25 seconds) ---
        # Slower frequencies for better tracking
        elif elapsed < 60.0:
            if elapsed < 8.1: self.get_logger().info('Phase 2: EXCITATION (Slow Figure-8)...')

            t_dance = elapsed - 8.0
            
            # --- Tuning Changes ---
            # Previous: sin(2.0 * t) -> Period ~3.14s
            # New:      sin(1.0 * t) -> Period ~6.28s (Twice as slow)
            
            # 1. TRANSLATION: The Slow Figure-8
            msg.linear.y = 1.0 * math.sin(1.0 * t_dance)  # Sway Left/Right (Slow)
            msg.linear.z = 0.5 * math.sin(2.0 * t_dance)  # Bob Up/Down (2x Sway freq)

            # 2. ROTATION: Gentle Banking
            # Reduced amplitude to prevent motion blur
            msg.angular.x = -0.2 * math.cos(1.0 * t_dance) # Roll
            msg.angular.y = 0.2 * math.sin(1.0 * t_dance) - 0.1 # Pitch (Tiny nod)
            msg.angular.z = -0.2 * math.cos(1.0 * t_dance) # Yaw

        # --- PHASE 3: FINISH ---
        else:
            if elapsed < 25.1: self.get_logger().info('Initialization Sequence Complete.')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.z = 0.0
            msg.angular.y = 0.0

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
        if rclpy.ok():
            node.publisher_.publish(Twist()) # Stop robot
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()