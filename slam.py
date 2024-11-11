import numpy as np
import cv2
from utils.data_management import Map
from utils.preprocessing import Preprocessing
from utils.triangulation import Triangulator
from utils.feature_handling import FeatureHandler
from bundle_adjustment import BundleAdjuster
from loop_closure import LoopClosureDetector
import traceback


class Slam:
    def __init__(self, data_path):
        self.map = Map()
        self.frame_count = 0
        self.preprocess = Preprocessing(data_path)
        self.orb = cv2.ORB_create(nfeatures=100)
        self.cam0_params = self.preprocess.get_sensor('cam0')
        self.cam1_params = self.preprocess.get_sensor('cam1')
        self.imu_params = self.preprocess.get_sensor('imu0')
        self.feature_handler = FeatureHandler(self.cam0_params)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bundle_adjuster = BundleAdjuster(self.cam0_params, self.cam1_params)
        self.triangulator = Triangulator(self.cam0_params, self.cam1_params)
        self.loop_closure_detector = LoopClosureDetector()
        self.bundle_adjust_frequency = 3


    def run(self):
        while True:
            try:
                self.process_frames()
                # Perform loop closure detection
                if self.loop_closure_detector.detect(self.map):
                    # Handle loop closure (e.g., update map, adjust poses)
                    pass

            except ValueError as e:
                print("End of dataset reached") 
                print(f"Error details: {str(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                break
            
    def get_cam_and_imu_data(self):
        camera_imgs = self.preprocess.get_image_frame(self.frame_count)
        cam0_img = cv2.imread(camera_imgs['cam0'], cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(camera_imgs['cam1'], cv2.IMREAD_GRAYSCALE)
        imu_data = self.preprocess.get_imu_data(self.frame_count)
        return cam0_img, cam1_img, imu_data

    def process_frames(self):
        print(self.frame_count)
        # Get data
        cam0_img, cam1_img, imu_data = self.get_cam_and_imu_data()
        
        # Extract features
        kp0, des0 = self.feature_handler.detect_features(cam0_img)
        kp1, des1 = self.feature_handler.detect_features(cam1_img)
        
        # Get pose and 3D points with matched features
        R, t, points_3d, pts0_inliers, pts1_inliers, P1, des0_inliers, des1_inliers = \
            self.get_R_t_and_pts(kp0, kp1, des0, des1)

        # Create matched feature data structure
        matched_features = {
            'points_3d': points_3d,
            'keypoints': {'cam0': pts0_inliers, 'cam1': pts1_inliers},
            'descriptors': {'cam0': des0_inliers, 'cam1': des1_inliers}
        }

        # Create keyframe with matched features
        self.map.create_keyframe(
            R=R, t=t, P=P1,
            matched_features=matched_features,
            imu_data=imu_data,
            frame_id=self.frame_count
        )

        if self.frame_count % self.bundle_adjust_frequency == 0 and self.frame_count != 0:
            self.bundle_adjuster.optimize(self.map)
        
        self.frame_count += 1

    def get_R_t_and_pts(self, kp0, kp1, des0, des1):
        # Get matched pairs
        matches = self.feature_handler.match_features(des0, des1)
        matched_pairs = [(m.queryIdx, m.trainIdx) for m in matches]
        des0_idx, des1_idx = zip(*matched_pairs)
        
        # Convert indices to numpy array for indexing
        des0_idx = np.array(des0_idx)
        des1_idx = np.array(des1_idx)
        
        # Get points and descriptors for matches
        pts0 = np.float32([kp0[idx].pt for idx in des0_idx])
        pts1 = np.float32([kp1[idx].pt for idx in des1_idx])
        
        # Get matched descriptors
        des0_matched = des0[des0_idx]  # Index once for matches
        des1_matched = des1[des1_idx]
        
        # Get inliers
        F, mask = self.feature_handler.compute_fundamental_matrix(pts0, pts1)
        mask = mask.ravel().astype(bool)
        
        # Apply mask to all arrays
        pts0_inliers = pts0[mask]
        pts1_inliers = pts1[mask]
        des0_inliers = des0_matched[mask]  # Apply mask to already matched descriptors
        des1_inliers = des1_matched[mask]
        
        E = self.feature_handler.compute_essential_matrix(F)
        delta_R, delta_t = self.feature_handler.decompose_essential_matrix(E, pts0_inliers, pts1_inliers)
        
        current_R = self.triangulator.current_R
        current_t = self.triangulator.current_t
        
        R = delta_R @ current_R
        t = delta_R @ current_t + delta_t
        
        points_3d, P1 = self.triangulator.triangulate_points(pts0_inliers, pts1_inliers, delta_R, delta_t)
        
        return R, t, points_3d, pts0_inliers, pts1_inliers, P1, des0_inliers, des1_inliers

    def get_map(self):
        return self.map
