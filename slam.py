import numpy as np
import cv2
from map_gen import Map
from preprocess import Preprocessing

class Slam:
    def __init__(self, preprocess):
        self.map = Map()
        self.frame_count = 0
        self.preprocess = preprocess
        self.orb = cv2.ORB_create(nfeatures=10000)
        self.cam0_params = self.preprocess.get_sensor('cam0')
        self.cam1_params = self.preprocess.get_sensor('cam1')
        self.imu_params = self.preprocess.get_sensor('imu0')
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def process_frames(self):
        camera_imgs = self.preprocess.get_image_frame(self.frame_count)
        cam0_img = cv2.imread(camera_imgs['cam0'], cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(camera_imgs['cam1'], cv2.IMREAD_GRAYSCALE)

        imu_data = self.preprocess.get_imu_data(self.frame_count)
        
        kp0, des0 = self.get_features(cam0_img)
        kp1, des1 = self.get_features(cam1_img)
        
        matches = self.match_features(des0, des1)
        
        F, mask = self.compute_fundamental_matrix(kp0, kp1, matches)
        E = self.compute_essential_matrix(F)
        
        R, t = self.decompose_essential_matrix(E)
        
        # Update map with new information
        self.map.update(kp0, kp1, R, t, imu_data)
        
        self.frame_count += 1
        
        return R, t

    def get_features(self, img):
        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)
        return np.array([kp.pt for kp in kp]), des

    def match_features(self, des0, des1):
        matches = self.bf.match(des0, des1)
        return sorted(matches, key=lambda x: x.distance)[:100]  # Keep top 100 matches

    def compute_fundamental_matrix(self, kp0, kp1, matches):
        pts0 = np.float32([kp0[m.queryIdx] for m in matches])
        pts1 = np.float32([kp1[m.trainIdx] for m in matches])
        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)
        return F, mask

    def compute_essential_matrix(self, F):
        K = np.array(self.cam0_params.extra_params['intrinsics']['parameters'])
        return K.T @ F @ K

    def decompose_essential_matrix(self, E):
        _, R, t, _ = cv2.recoverPose(E)
        return R, t

    def get_map(self):
        return self.map

    def generate_map(self):
        # Implement map generation logic here
        # This could involve creating a 3D point cloud from the tracked features
        pass

    def run(self):
        while True:
            try:
                R, t = self.process_frames()
                # Perform loop closure detection here
                # Perform bundle adjustment here
                print(f"Frame {self.frame_count}: Rotation = {R}, Translation = {t}")
            except ValueError:
                # End of dataset reached
                break
            
            self.generate_map()
