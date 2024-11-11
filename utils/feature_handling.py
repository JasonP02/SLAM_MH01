import cv2
import numpy as np

class FeatureHandler:
    def __init__(self, cam0_params):
        self.orb = cv2.ORB_create(nfeatures=100)
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 
            dict(checks=50)
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K_raw = np.array(cam0_params.extra_params['intrinsics'])
        self.K = np.array([[self.K_raw[0], 0, self.K_raw[2]],
                      [0, self.K_raw[1], self.K_raw[3]],
                      [0, 0, 1]])

    def detect_features(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_features(self, des1, des2):
        matches = self.bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)[:100]

    def compute_fundamental_matrix(self, pts0, pts1):
        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)
        if mask is None:
            mask = np.ones(len(pts0), dtype=bool)
        return F, mask

    def compute_essential_matrix(self, F):
        return self.K.T @ F @ self.K

    def decompose_essential_matrix(self, E, pts0_inliers, pts1_inliers):
        _, R, t, _ = cv2.recoverPose(E, pts0_inliers, pts1_inliers, self.K)
        return R, t

