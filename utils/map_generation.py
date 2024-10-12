import numpy as np

class Map():
    def __init__(self, max_features=1000):
        self.points_3d = np.zeros((3, max_features))
        self.keyframes = []

    def update(self, R, t, P, cam0, cam1, imu_data, frame_count):
        # Create a new Features object
        new_keyframe = Keyframe(R, t, P, cam0, cam1, imu_data, frame_count)
        self.add_keyframe(new_keyframe)

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)

    def remove_keyframe(self, index):
        if 0 <= index < len(self.keyframes):
            del self.keyframes[index]

class Keyframe():
    def __init__(self, R, t, P, cam0, cam1, imu_data, frame_count):
        self.R = R  # Rotation matrix
        self.t = t  # Translation vector
        self.P = P
        self.cam0_kp = cam0['kp']
        self.cam1_kp = cam1['kp']
        self.cam0_des = cam0['des']
        self.cam1_des = cam1['des']
        self.imu_data = imu_data
        self.frame_count = frame_count  # Number of frames


class MapPoint():
    def __init__(self):
        self.point3d = np.array(3,1)
        self.frames_that_contain = [] # List of frames that contain a 2d representation of this point (!!)
