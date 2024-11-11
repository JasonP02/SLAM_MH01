import numpy as np
import cv2
import numpy as np

class Map():
    def __init__(self):
        self.keyframes = []
        self.point_tracker = PointTracker()

    def create_keyframe(self, R, t, P, matched_features, imu_data, frame_id):
        map_points = []
        points_3d = matched_features['points_3d']
        kp0 = matched_features['keypoints']['cam0']
        kp1 = matched_features['keypoints']['cam1']
        des0 = matched_features['descriptors']['cam0']
        des1 = matched_features['descriptors']['cam1']

        # Process all matched features at once
        for idx in range(len(points_3d)):
            map_point = self.point_tracker.create_point(
                pt3d=points_3d[idx],
                frame=frame_id,
                idx=idx,
                descriptors=[des0[idx], des1[idx]],
                keypoints=[kp0[idx], kp1[idx]]
            )
            map_points.append(map_point)
            
        keyframe = Keyframe(R, t, P, matched_features, imu_data, frame_id, map_points)
        self.keyframes.append(keyframe)
        return keyframe

class Keyframe():
    def __init__(self, R, t, P, matched_features, imu_data, frame_id, map_points):
        self.R = R
        self.t = t
        self.P = P
        self.keypoints = matched_features['keypoints']
        self.descriptors = matched_features['descriptors']
        self.imu_data = imu_data
        self.id = frame_id
        self.map_points = map_points


class PointTracker:
    def __init__(self):
        self._next_point_id = 0
        self.active_points = {}  # id -> MapPoint
        self.descriptor_match_threshold = 50
        
    def find_matching_point(self, descriptors):
        """Find existing point with similar descriptor using either camera view"""
        min_distance = float('inf')
        best_match = None
        
        for point in self.active_points.values():
            # Try matching with both descriptors
            for desc in descriptors:
                for stored_desc in point.descriptors:
                    distance = cv2.norm(desc, stored_desc, cv2.NORM_HAMMING)
                    if distance < min_distance and distance < self.descriptor_match_threshold:
                        min_distance = distance
                        best_match = point
                
        return best_match
        
    def create_point(self, pt3d, frame, idx, descriptors, keypoints):
        # Check if this point matches an existing one using either descriptor
        existing_point = self.find_matching_point(descriptors)
        
        if existing_point is not None:
            # Add new observations to existing point for both cameras
            existing_point.add_observation(frame, idx, keypoints, descriptors)
            return existing_point
        else:
            # Create new point if no match found
            point = MapPoint(pt3d, frame, idx, self._next_point_id, descriptors, keypoints)
            self.active_points[self._next_point_id] = point
            self._next_point_id += 1
            return point

class MapPoint:
    def __init__(self, pt3d, frame, idx, point_id, descriptors, keypoints):
        self.id = point_id
        self.position = pt3d
        self.descriptors = descriptors  # List of descriptors from different views
        self.observations = []  # List of (frame_id, feature_idx, keypoint, camera_id)
        self.add_observation(frame, idx, keypoints, descriptors)
        
    def add_observation(self, frame_id, feature_idx, keypoints, descriptors):
        """Add observations from both cameras"""
        for cam_id, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            self.observations.append((frame_id, feature_idx, kp, cam_id))
            if desc is not None:  # Update descriptors if new one is provided
                self.descriptors[cam_id] = desc