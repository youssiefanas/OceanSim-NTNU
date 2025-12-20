# OceanSim - Underwater Robotics Simulator for Isaac Sim

OceanSim is a comprehensive extension for NVIDIA Isaac Sim designed to simulate underwater environments and robotic systems. It provides realistic sensor simulation, ROS2 integration, and tools for large-scale synthetic data generation.

## Features

*   **Realistic Underwater Sensors**:
    *   **Underwater Camera**: Simulates light attenuation, backscatter, and particulates. Configurable via YAML.
    *   **Imaging Sonar**: Simulates acoustic imaging (Visual only).
    *   **DVL (Doppler Velocity Log)**: Provides velocity estimates with configurable noise and dropout simulation.
    *   **IMU**: Accelerometer and Gyroscope with configurable bias and noise models.
    *   **Barometer**: Depth/Pressure simulation based on hydrostatic pressure.
*   **ROS2 Integration**: Full bridge support. Publishes sensor data to ROS2 topics and accepts control commands.
*   **Data Collection System**: Automated logging of synchronized sensor data and ground truth trajectories for machine learning datasets.
*   **Random Trajectory Generation**: Integrated path planner (A*) that generates random collision-free paths using 2D occupancy maps.
*   **Configurable Environment**: Externalized configuration for generic assets, maps, and simulation parameters.

## Configuration

OceanSim uses YAML files to manage paths and settings, allowing flexibility without changing code.

### 1. Main Configuration (`config/config.yaml`)
Defines the paths to commonly used assets, maps, and global settings.
*   **`paths`**: Locations of USD assets (robot, environment items) and map config files.
*   **`filenames`**: Default filenames for output or loading (e.g., `map.yaml`, `imu_metadata.yaml`).

### 2. Map Configuration (`map.yaml`)
Defines the 2D occupancy grid used for navigation and random path generation.
*   **image**: Path to the PGM/PNG occupancy image.
*   **resolution**: Meters per pixel.
*   **origin**: Pose of the map origin in the world frame.

> **Note on Map Generation**: The occupancy map logic is adapted from the [Isaac Sim Replicator Mobility Gen tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/synthetic_data_generation/tutorial_replicator_mobility_gen.html). We are *not* using the `mobility_gen` extension directly to avoid conflicts, but have adapted its core algorithms for OceanSim.

### 3. Underwater Camera Config
YAML files (e.g., `Jerlov_II.yaml`) controlling visual rendering:
*   `atten_coeff`: RGB attenuation coefficients.
*   `backscatter_coeff`: RGB scattering coefficients.
*   `backscatter_value`: Ambient light color/intensity.

## Usage Guide

### 1. Launching the Simulator
1.  Open Isaac Sim.
2.  Enable the `OceanSim-NTNU` extension in the Extension Manager.
3.  The **Sensor Example** window will appear.

### 2. User Interface Controls

**Setup (Do this BEFORE clicking LOAD):**
*   **Sensors**: Use checkboxes to enable/disable specific sensors (Sonar, Camera, DVL, Barometer, IMU).
    *   *Note: Using the Camera requires checking the generic "Camera" box. You can then provide a specific UW Config YAML file path (Optional).*
*   **Path to USD**: (Optional) specific USD environment to load. Leave empty for default.
*   **Control Mode**:
    *   `Manual control`: Use Keyboard or **Gamepad/Joystick**.
    *   `ROS control`: Robot listens to `/cmd_vel` or force topics.
    *   `Waypoints`: Robot follows a pre-defined path or a **randomly generated path** if enabled.

**Run Scenario:**
1.  **LOAD**: Initializes the robot, sensors, and environment.
2.  **RESET**: **Important:** Always click RESET after LOAD to ensure proper sensor initialization.
3.  **RUN/STOP**: Starts or pauses the physics simulation.

### 3. Data Collection Mode
To generate synthetic datasets:
1.  Check **"Data Collection Mode"**.
2.  Select a **"Path to save data"**.
3.  Click **LOAD** and then **RUN**.
4.  The simulator will record data from all enabled sensors.

**Output Structure:**
The system creates a timestamped folder or uses your selected folder to save:
*   `camera_sensor/`: RGB Images and Depth `.npy` files.
*   `DVL_sensor/dvl_data.csv`: Timestamp, Velocity (u, v, w).
*   `barometer_sensor/barometer_data.csv`: Timestamp, Pressure (Pa).
*   `IMU_sensor/imu_data.csv`: Timestamp, Accel(xyz), Gyro(xyz).
*   `ground_truth/trajectory.csv`: Timestamp, Position(xyz), Orientation(quat).

## ROS2 Interface

OceanSim automatically publishes sensor data to ROS2 topics if the bridge is enabled.

| Sensor | Topic | Message Type |
| :--- | :--- | :--- |
| **Camera** | `/oceansim/robot/uw_img` | `sensor_msgs/CompressedImage` |
| **DVL** | `/dvl/velocity` | `geometry_msgs/TwistStamped` |
| **Barometer** | `/barometer/pressure` | `sensor_msgs/FluidPressure` |
| **IMU** | `/oceansim/robot/imu` | `sensor_msgs/Imu` |
| **Pose (GT)**| `/oceansim/robot/pose` | `geometry_msgs/PoseStamped` |
| **CMD_Vel** | `/oceansim/robot/cmd_vel` | `geometry_msgs/Twist` (Subscriber) |

## Contributing

This project is open-source. Pull requests are welcome for new sensors, improved physical models, or additional environments.