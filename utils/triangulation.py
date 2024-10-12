import cv2
import numpy as np

class Triangulator:
    def __init__(self, cam0_params, cam1_params):
        self.K0 = np.array(cam0_params['intrinsics']['parameters'])
        self.K1 = np.array(cam1_params['intrinsics']['parameters'])
        self.baseline = np.linalg.norm(cam1_params['T_cn_cnm1'][:3, 3])
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))

    def update_pose(self, R, t):
        self.current_R = R @ self.current_R
        self.current_t = R @ self.current_t + t

    def triangulate_points(self, kp0, kp1, R, t):
        P0 = self.K0 @ self.get_projection_matrix(self.current_R, self.current_t)
        P1 = self.K1 @ self.get_projection_matrix(R @ self.current_R, R @ self.current_t + t)

        points_4d = cv2.triangulatePoints(P0, P1, kp0.T, kp1.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return points_3d.T

    def get_projection_matrix(self, R, t):
        return np.hstack((R, t))
