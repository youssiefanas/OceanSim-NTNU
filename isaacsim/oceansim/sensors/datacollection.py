import os

class DataCollectionSensor:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def collect_data(self, name: str):
        # Placeholder for data collection logic
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.name = name
        sensor_path = os.path.join(self.data_path, self.name)
        if not os.path.exists(sensor_path):
            os.makedirs(sensor_path)
        print(f"Collecting data from sensor '{self.name}' at '{self.data_path}'")
        
        return sensor_path

        