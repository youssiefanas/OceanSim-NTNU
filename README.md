# BlueROV with IMU

This guide will help you set up and run the BlueROV simulation with an IMU sensor in OceanSim.

### 1. Download the BlueROV Asset File

Download the `BROV_IMU.usd` file from:
[Google Drive Link](https://drive.google.com/file/d/1iUSkD-w_9yzr1Q_m8cNrHmGdzrtYz2bs/view?usp=sharing)

### 2. Place the Asset File

Save the downloaded file in your OceanSim assets directory:

```text
OceanSim_assets/Bluerov/BROV_IMU.usd
```

### 3. Register Asset Path

If you haven't already configured OceanSim to recognize your asset path, run:

```bash
cd /path/to/OceanSim
python3 config/register_asset_path.py /path/to/OceanSim_assets
```

### 4. [Optional] Branch Configuration

If you're working from your own branch instead of this one, ensure the USD path is correctly set:

- Open `isaacsim/oceansim/modules/SensorExample_python/ui_builder.py`

- Verify this line points to your file:

```python
robot_usd_path = get_oceansim_assets_path() + "/Bluerov/BROV_IMU.usd"
```

**Alternatively:** Simply rename your downloaded file to `BROV_low.usd` to match the default configuration.

## Running the Simulation

### 5. Launch IsaacSim with OceanSim

1. Start Isaac Sim with the OceanSim extension

2. Open the Sensor Example tab

3. Select the additional sensors you want to include

4. Click LOAD

> **Note**: No need to manually select a USD path—this was automatically handled when you registered the asset path.

## Troubleshooting

### 6. Giant Camera

If a large camera appears in the simulation:

- Toggle off the visibility property from the UW_camera prim in the scene hierarchy.

### 7. IMU Data Not Publishing

If you encounter this warning in the console or Action Graph:

```text
2025-10-24 21:42:46 [Warning] [omni.graph.core.plugin] /World/rob/IMUActionGraph/isaac_read_imu_node: [/World/rob/IMUActionGraph] no valid sensor reading, is the sensor enabled?
```

**Solution:**

1. Click the **RESET** button in the Sensor Example tab

2. Run the simulation again

## Data Visualization with Foxglove

### 8. Install Foxglove Bridge

```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```

### 9. Launch Foxglove Bridge
```
ros2 run foxglove_bridge foxglove_bridge
```

### 10. Connect and Visualize

1. Open Foxglove Studio in your browser

2. Establish a new connection

3. Select the appropriate WebSocket connection

4. You should now be able to visualize the IMU data and other ROS 2 topics