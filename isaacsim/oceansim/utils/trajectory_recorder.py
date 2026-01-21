import numpy as np
import os
import csv

class TrajectoryRecorder:
    def __init__(self, filepath, mode='record'):
        """
        Initialize the TrajectoryRecorder.
        
        Args:
            filepath (str): Path to the file for saving/loading data.
            mode (str): 'record' to save data, 'replay' to load data.
        """
        self.filepath = filepath
        self.mode = mode
        self.data = []
        self._playback_index = 0
        
        if self.mode == 'replay':
            self.load()

    def record(self, timestamp, force, torque):
        """
        Record a single data point.
        
        Args:
            timestamp (float): Simulation time.
            force (np.array): Force vector [fx, fy, fz].
            torque (np.array): Torque vector [tx, ty, tz].
        """
        if self.mode != 'record':
            return
            
        row = [timestamp] + list(force) + list(torque)
        self.data.append(row)

    def save(self):
        """Save the recorded data to a CSV file."""
        if self.mode != 'record' or not self.data:
            return

        # Ensure directory exists
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['time', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
                writer.writerows(self.data)
            print(f"[TrajectoryRecorder] Saved {len(self.data)} points to {self.filepath}")
        except Exception as e:
            print(f"[TrajectoryRecorder] Error saving trajectory: {e}")

    def load(self):
        """Load data from the CSV file for replay."""
        if not os.path.exists(self.filepath):
            print(f"[TrajectoryRecorder] File not found: {self.filepath}")
            return

        try:
            with open(self.filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                self.data = []
                for row in reader:
                    # Convert strings to floats
                    self.data.append([float(x) for x in row])
            print(f"[TrajectoryRecorder] Loaded {len(self.data)} points from {self.filepath}")
        except Exception as e:
            print(f"[TrajectoryRecorder] Error loading trajectory: {e}")

    def get_command(self, timestamp):
        """
        Get the force and torque commands for the given timestamp.
        Uses simple nearest neighbor or linear interpolation. 
        For now, let's just find the closest point or the next point in sequence.
        Since simulation steps are generally consistent, iterating through the list is efficient.
        """
        if self.mode != 'replay' or not self.data:
             # Default zero command
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        # Find the row that corresponds to the current time
        # We assume time is monotonically increasing.
        # We search from the last known index.
        
        while self._playback_index < len(self.data) - 1:
            # Check if the next point's time is closer to the requested timestamp than the current one
            current_t = self.data[self._playback_index][0]
            next_t = self.data[self._playback_index + 1][0]
            
            if abs(timestamp - next_t) < abs(timestamp - current_t):
                self._playback_index += 1
            else:
                break
        
        # Get values
        row = self.data[self._playback_index]
        # row structure: [time, fx, fy, fz, tx, ty, tz]
        force = np.array(row[1:4])
        torque = np.array(row[4:7])
        
        return force, torque
