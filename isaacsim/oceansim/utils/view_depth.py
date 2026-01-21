import numpy as np
import matplotlib.pyplot as plt
import os

# REPLACE THIS with the path to one of your saved .npy files
file_path = "/home/youssief/coral_reefdata_collected_oceansim/camera_sensor/Depth/Depth_40.npy"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # 1. Load the raw depth data
    # This array contains the distance in meters for every pixel
    depth_map = np.load(file_path)

    print(f"Depth Map Shape: {depth_map.shape}")
    print(f"Min Distance: {np.nanmin(depth_map):.2f} m")
    print(f"Max Distance: {np.nanmax(depth_map):.2f} m")

    # 2. Handle 'Infinite' depths (e.g., the sky)
    # Replicator often returns 'inf' for the sky. We replace it with the 
    # maximum valid depth so the plot doesn't break or look all one color.
    # Alternatively, you can set it to 0 or NaN.
    max_valid_depth = np.nanmax(depth_map[depth_map != np.inf])
    depth_map[depth_map == np.inf] = max_valid_depth

    # 3. Plot the heatmap
    # make size big enough to see the image and font for the title and axis labels
    plt.figure(figsize=(20, 12))
    
    # 'viridis' or 'plasma' are good colormaps for depth (Yellow=Far, Purple=Close)
    # cmap = plt.cm.get_cmap("jet").copy()
    plt.imshow(depth_map, cmap='viridis') 
    
    cbar = plt.colorbar()
    cbar.set_label('Distance (meters)', size=16)
    cbar.ax.tick_params(labelsize=14) 

    plt.title(f"Depth Map Visualization\n{os.path.basename(file_path)}", fontsize=20)
    plt.axis('off') # Hide axis ticks
    plt.savefig("depth_map_" + os.path.basename(file_path) + ".png")
    plt.show()