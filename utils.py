class Map():
    def __init__(self):
        pass

    def add_keyframe(self):
        pass

    def remove_keyframe(self):
        pass 

class Features():
    def __init__(self, R, t, frame_count, num_features, feature_locations, P):
        self.R = R
        self.t = t
        self.frame_count = frame_count
        self.num_features = num_features
        self.feature_locations = feature_locations
        self.P = P

class MapPoint():
    def __init__(self):
        self.3d_points = np.