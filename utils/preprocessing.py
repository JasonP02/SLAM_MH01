import os
import csv
import glob
import numpy as np
import yaml

# General class for sensor parameters
class SensorParams:
    def __init__(self, sensor_type, comment, T_BS, rate_hz, **kwargs):
        self.sensor_type = sensor_type
        self.comment = comment
        self.T_BS = np.array(T_BS).reshape(4, 4)
        self.rate_hz = rate_hz
        
        # Handle optional keyword arguments for sensor-specific params
        self.extra_params = kwargs

    def __repr__(self):
        return f"{self.sensor_type} ({self.comment}), rate: {self.rate_hz} Hz"

class Preprocessing:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cam0_path = os.path.join(self.root_dir, 'cam0')
        self.cam1_path = os.path.join(self.root_dir, 'cam1')
        self.imu0_path = os.path.join(self.root_dir, 'imu0')
        
        # Cache image files
        self.cam0_path_files = sorted(glob.glob(os.path.join(self.cam0_path, 'data', '*.png')), reverse=True)
        self.cam1_path_files = sorted(glob.glob(os.path.join(self.cam1_path, 'data', '*.png')), reverse=True)
        
        # Cache IMU data
        self.imu_data = self._load_imu_data()
        
        # Load all sensor parameters into a dictionary
        self.sensors = {
            'cam0': self._load_sensor_params(self.cam0_path, 'sensor.yaml'),
            'cam1': self._load_sensor_params(self.cam1_path, 'sensor.yaml'),
            'imu0': self._load_sensor_params(self.imu0_path, 'sensor.yaml')
        }

    def _load_imu_data(self):
        imu_data_path = os.path.join(self.imu0_path, 'data.csv')
        if not os.path.exists(imu_data_path):
            raise FileNotFoundError(f"IMU data file not found: {imu_data_path}")
        
        with open(imu_data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            return np.array([row for row in reader], dtype=float)

    def _load_sensor_params(self, sensor_path, yaml_file):
        params_path = os.path.join(sensor_path, yaml_file)
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Sensor config file not found: {params_path}")
        
        with open(params_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Extract common and specific fields
        sensor_type = data['sensor_type']
        comment = data.get('comment', 'No comment')
        T_BS = data['T_BS']['data']
        rate_hz = data['rate_hz']
        
        # Pass any extra fields as kwargs for sensor-specific parameters
        extra_params = {k: v for k, v in data.items() if k not in ['sensor_type', 'comment', 'T_BS', 'rate_hz']}
        
        return SensorParams(sensor_type, comment, T_BS, rate_hz, **extra_params)

    def get_image_frame(self, frame_num):
        if frame_num < 0 or frame_num >= len(self.cam0_path_files) or frame_num >= len(self.cam1_path_files):
            raise ValueError("Invalid frame number")
        
        return {
            'cam0': self.cam0_path_files[frame_num],
            'cam1': self.cam1_path_files[frame_num]
        }

    def get_sensor(self, sensor_name):
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor '{sensor_name}' not found")
        return self.sensors[sensor_name]

    def get_imu_data(self, frame_num):
        start_idx = frame_num * 10
        end_idx = start_idx + 10
        
        if start_idx < 0 or end_idx > len(self.imu_data):
            raise ValueError("Invalid frame number")
        
        return self.imu_data[start_idx:end_idx, 1:]  # Exclude timestamp column