import cv2
import numpy as np

class FeatureHandler:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=10000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, des1, des2):
        matches = self.bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)[:100]

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
