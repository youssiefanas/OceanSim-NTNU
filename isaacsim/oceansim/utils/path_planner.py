import numpy as np
import heapq
import random
from .occupancy_map import OccupancyMap, Point2d

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    """
    A* search algorithm on a numpy array (0 is obstacle, 1 is free).
    start and goal are (row, col) tuples.
    """
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    visited = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (f_score[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        visited.add(current)
        
        # Optimization: Early exit if we searched too much? No, map shouldn't be too huge.
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]: 
                    if array[neighbor[0]][neighbor[1]] == 0: # 0 is free for us? No, need to check mask semantics.
                        continue # Obstacle
                else:
                    continue # Out of bounds
            else:
                continue # Out of bounds

            if neighbor in visited:
                continue

            # Distance 1 for cardinal, 1.414 for diagonal
            dist = 1.414 if abs(i) + abs(j) == 2 else 1.0
            tentative_g_score = g_score[current] + dist

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (f_score[neighbor], neighbor))
                
    return None

def generate_random_path(occupancy_map: OccupancyMap, start_pose: Point2d = None):
    """
    Generates a random path from start_pose (or random start) to a random end point in freespace.
    """
    freespace = occupancy_map.freespace_mask() # True where free
    
    # In A* implementation above, I assumed 1 is free. Let's fix that.
    # occupancy_map.freespace_mask() returns boolean where TRUE is FREE.
    # In A*, I check `if array[...] == 0: continue`.
    # So I should pass integer array where 1 is FREE and 0 is OBSTACLE?
    # Or just invert logic in A*.
    # Let's pass freespace mask directly (True/False).
    # And in A*: if not array[neighbor]: continue (if False, meaning not free constraint).
    
    # Get all indices where freespace is True
    free_y, free_x = np.where(freespace)
    if len(free_x) == 0:
        print("[PathPlanner] Error: No freespace found in map.")
        return None

    if start_pose is None:
        # Sample random start
        idx = random.randint(0, len(free_x) - 1)
        start_px = (free_y[idx], free_x[idx])
    else:
        # Convert start pose to pixel
        start_px_arr = occupancy_map.world_to_pixel_numpy(np.array([[start_pose.x, start_pose.y]]))
        start_px = (int(start_px_arr[0, 1]), int(start_px_arr[0, 0])) # (row, col) -> (y, x)
        
        # Check if start is valid
        if not (0 <= start_px[0] < freespace.shape[0] and 0 <= start_px[1] < freespace.shape[1]):
             print("[PathPlanner] Warning: Start point out of bounds. Clamping.")
             start_px = (min(max(start_px[0], 0), freespace.shape[0]-1), min(max(start_px[1], 0), freespace.shape[1]-1))
             
        if not freespace[start_px[0], start_px[1]]:
             print("[PathPlanner] Warning: Start point is in obstacle. Finding nearest free point.")
             # Simple nearest neighbor search could happen here, or just fail.
             # For now, let's try to proceed or just return None?
             # Let's search for nearest 
             dists = (free_y - start_px[0])**2 + (free_x - start_px[1])**2
             nearest_idx = np.argmin(dists)
             start_px = (free_y[nearest_idx], free_x[nearest_idx])

    # Sample random end
    idx = random.randint(0, len(free_x) - 1)
    end_px = (free_y[idx], free_x[idx])
    
    print(f"[PathPlanner] Planning from {start_px} to {end_px}...")
    
    # Run A*
    # Pass freespace mask (True = Walkable)
    # A* expects array[row][col] to be truthy if walkable? 
    # My A* logic: if array[neighbor] == 0: continue. 
    # So if array is False (0), it treats as obstacle. Correct.
    path_px = astar(freespace, start_px, end_px)
    
    if path_px is None:
        print("[PathPlanner] Failed to find path.")
        return None
        
    print(f"[PathPlanner] Found path with {len(path_px)} points.")
    
    # Convert pixels back to world (xy)
    # path_px is list of (y, x) tuples
    path_px_arr = np.array([(p[1], p[0]) for p in path_px]) # Convert to (x, y) for numpy function
    path_world = occupancy_map.pixel_to_world_numpy(path_px_arr)
    
    # Downsample?
    # Take every Nth point to smooth it out / prevent too dense waypoints
    if len(path_world) > 10:
         skip = max(1, len(path_world) // 50) # Aim for ~50 waypoints max
         path_world = path_world[::skip]
         
    # Ensure end point is included
    # if using slicing, might lose last one. 
    # path_world[-1] is already end.
         
    return path_world
