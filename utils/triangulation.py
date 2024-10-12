import cv2
import numpy as np

class Triangulator:
    def __init__(self, cam0_params, cam1_params):
        self.K0 = np.array(cam0_params['intrinsics']['parameters'])
        self.K1 = np.array(cam1_params['intrinsics']['parameters'])
        self.baseline = np.linalg.norm(cam1_params['T_cn_cnm1'][:3, 3])

    def triangulate_points(self, kp0, kp1, R, t):
        P0 = self.K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = self.K1 @ np.hstack((R, t))

        points_4d = cv2.triangulatePoints(P0, P1, kp0.T, kp1.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return points_3d.T

    def get_projection_matrix(self, R, t):
        return np.hstack((R, t))