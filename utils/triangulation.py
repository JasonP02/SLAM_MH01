import cv2
import numpy as np

class Triangulator:
    def __init__(self, cam0_params, cam1_params):
        K0_raw = np.array(cam0_params.extra_params['intrinsics'])
        K1_raw = np.array(cam1_params.extra_params['intrinsics'])
        
        # Reshape camera matrices to 3x3
        self.K0 = np.array([[K0_raw[0], 0, K0_raw[2]],
                           [0, K0_raw[1], K0_raw[3]],
                           [0, 0, 1]])
        self.K1 = np.array([[K1_raw[0], 0, K1_raw[2]],
                           [0, K1_raw[1], K1_raw[3]],
                           [0, 0, 1]])
                           
        self.baseline = np.linalg.norm(cam1_params.T_BS[:3, 3])
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))

    def update_pose(self, R, t):
        # Ensure R is 3x3
        if R.shape != (3, 3):
            raise ValueError(f"Expected R shape (3,3), got {R.shape}")
        
        # Ensure t is 3x1
        if t.shape != (3, 1):
            t = t.reshape(3, 1)
        
        # Update pose
        self.current_R = R @ self.current_R
        self.current_t = R @ self.current_t + t

    def triangulate_points(self, kp0, kp1, R, t):
        P0 = self.K0 @ self.get_projection_matrix(self.current_R, self.current_t)
        P1 = self.K1 @ self.get_projection_matrix(R, t)

        points_4d = cv2.triangulatePoints(P0, P1, kp0.T, kp1.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return points_3d.T, P1

    def get_projection_matrix(self, R, t):
        # Ensure t is 3x1
        if t.shape != (3, 1):
            t = t.reshape(3, 1)
        # Create 3x4 projection matrix [R|t]
        return np.hstack((R, t))
