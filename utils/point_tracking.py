import cv2
import numpy as np

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