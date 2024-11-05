import numpy as np
import cv2
from utils.map_generation import Map
from utils.preprocessing import Preprocessing
from utils.triangulation import Triangulator
from utils.feature_handling import FeatureHandler
from bundle_adjustment import BundleAdjuster
from loop_closure import LoopClosureDetector

class Slam:
    def __init__(self, data_path):
        self.map = Map()
        self.frame_count = 0
        self.preprocess = Preprocessing(data_path)
        self.orb = cv2.ORB_create(nfeatures=10000)
        self.cam0_params = self.preprocess.get_sensor('cam0')
        self.cam1_params = self.preprocess.get_sensor('cam1')
        self.imu_params = self.preprocess.get_sensor('imu0')
        self.feature_handler = FeatureHandler
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bundle_adjuster = BundleAdjuster()
        self.triangulator = Triangulator(self.cam0_params.extra_params, self.cam1_params.extra_params)
        self.loop_closure_detector = LoopClosureDetector()
        self.bundle_adjust_frequency = 10  # Adjust this value as needed

         
    def get_cam_and_imu_data(self):
        # Get our data
        camera_imgs = self.preprocess.get_image_frame(self.frame_count)
        cam0_img = cv2.imread(camera_imgs['cam0'], cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(camera_imgs['cam1'], cv2.IMREAD_GRAYSCALE)
        imu_data = self.preprocess.get_imu_data(self.frame_count)
        return cam0_img, cam1_img, imu_data

    def process_frames(self):
        cam0_img, cam1_img, imu_data = self.get_cam_and_imu_data()
        kp0, des0 = self.feature_handler.detect_and_compute(cam0_img)
        kp1, des1 = self.feature_handler.detect_and_compute(cam1_img)
        
        # Returns what we need for bundle adjustment, etc.
        R, t, points_3d, pts0, pts1 = self.get_R_t_and_pts(kp0, kp1, des0, des1)

        if self.frame_count % self.bundle_adjust_frequency == 0:
            self.bundle_adjuster.update(self.map, R, t, points_3d)

        # Update map with new information
        self.map.update(pts0, pts1, R, t, points_3d, imu_data, self.frame_count)
        
        # Update the triangulator's pose
        self.triangulator.update_pose(R, t)
        self.frame_count += 1

    def get_map(self):
        return self.map

    def get_R_t_and_pts(self, kp0, kp1, des0, des1):
        matches = self.feature_handler.match_features(des0, des1)
        
        # Use matched keypoints for fundamental matrix computation
        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        
        F, mask = self.feature_handler.compute_fundamental_matrix(pts0, pts1)
        
        # Use the mask to filter out outliers
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
        
        E = self.feature_handler.compute_essential_matrix(F)
        delta_R, delta_t = self.feature_handler.decompose_essential_matrix(E)

        # Get the current cumulative transformation
        current_R = self.triangulator.current_R
        current_t = self.triangulator.current_t
        
        # Compute the new absolute transformation
        R = delta_R @ current_R
        t = delta_R @ current_t + delta_t

        points_3d = self.triangulator.triangulate_points(pts0, pts1, delta_R, delta_t)

        return R, t, points_3d, pts0, pts1

    def run(self):
        while True:
            try:
                self.process_frames()
                # Perform loop closure detection
                if self.loop_closure_detector.detect(self.map):
                    # Handle loop closure (e.g., update map, adjust poses)
                    pass

            except ValueError:
                print("End of dataset reached") 
                break
            
