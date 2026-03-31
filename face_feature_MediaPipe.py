"""
面部特征点检测组件
提供高鲁棒性的面部特征和视线特征提取。
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Any

class FaceLandmarkDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Any:
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        frame.flags.writeable = True
        return results

    def extract_comprehensive_features(self, results: Any, image_shape: Tuple) -> Optional[dict]:
        """
        提取用于视线追踪的综合特征向量，包含头部姿态和基于局部坐标系的眼球特征。
        """
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h_img, w = image_shape[:2]

        # 1. 估计头部姿态
        head_pose = self._estimate_head_pose(face_landmarks, (h_img, w))

        # 2. 提取眼部局部相对特征 (基于投影法)
        def get_relative_iris_pos(left_idx, right_idx, top_idx, bottom_idx, iris_idx):
            """
            建立局部坐标系并投影虹膜位置。
            参数传入的 left 和 right 是指屏幕上的左和右，方便统一 X 轴方向。
            """
            p_left = np.array([face_landmarks.landmark[left_idx].x * w, face_landmarks.landmark[left_idx].y * h_img])
            p_right = np.array([face_landmarks.landmark[right_idx].x * w, face_landmarks.landmark[right_idx].y * h_img])
            p_top = np.array([face_landmarks.landmark[top_idx].x * w, face_landmarks.landmark[top_idx].y * h_img])
            p_bottom = np.array([face_landmarks.landmark[bottom_idx].x * w, face_landmarks.landmark[bottom_idx].y * h_img])
            iris = np.array([face_landmarks.landmark[iris_idx].x * w, face_landmarks.landmark[iris_idx].y * h_img])

            # 计算眼睛的中心点
            center = (p_left + p_right) / 2.0

            # 定义局部 X 轴 (向右) 和 Y 轴 (向下)
            vec_x = p_right - p_left
            vec_y = p_bottom - p_top

            width = np.linalg.norm(vec_x)
            height = np.linalg.norm(vec_y)

            if width < 1e-5 or height < 1e-5:
                return 0.0, 0.0

            dir_x = vec_x / width
            dir_y = vec_y / height

            # 虹膜偏离中心的向量
            vec_iris = iris - center

            # 投影到 X 和 Y 轴上，并归一化到 [-1, 1] 区间
            # 除以 (width/2) 是因为从中心到边缘的距离是宽度的一半
            rel_x = np.dot(vec_iris, dir_x) / (width / 2.0)
            rel_y = np.dot(vec_iris, dir_y) / (height / 2.0)

            return float(rel_x), float(rel_y)

        if len(face_landmarks.landmark) > 473:
            # 屏幕右侧的眼睛 (用户的左眼): 左眼角133, 右眼角33, 上159, 下145, 虹膜468
            l_iris_rel_x, l_iris_rel_y = get_relative_iris_pos(133, 33, 159, 145, 468)
            # 屏幕左侧的眼睛 (用户的右眼): 左眼角263, 右眼角362, 上386, 下374, 虹膜473
            r_iris_rel_x, r_iris_rel_y = get_relative_iris_pos(263, 362, 386, 374, 473)
        else:
            l_iris_rel_x, l_iris_rel_y, r_iris_rel_x, r_iris_rel_y = 0.0, 0.0, 0.0, 0.0

        return {
            "head_pitch": head_pose["pitch"],
            "head_yaw": head_pose["yaw"],
            "head_roll": head_pose["roll"],
            "head_tx": head_pose["tvec"][0][0],
            "head_ty": head_pose["tvec"][1][0],
            "head_tz": head_pose["tvec"][2][0],
            "l_iris_rel_x": l_iris_rel_x,
            "l_iris_rel_y": l_iris_rel_y,
            "r_iris_rel_x": r_iris_rel_x,
            "r_iris_rel_y": r_iris_rel_y,
            "rvec": head_pose["rvec"],
            "tvec": head_pose["tvec"]
        }

    def _estimate_head_pose(self, face_landmarks: Any, image_size: Tuple[int, int]) -> dict:
        h, w = image_size
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        return {
            "pitch": np.degrees(x),
            "yaw": np.degrees(y),
            "roll": np.degrees(z),
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs
        }

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        if not results.multi_face_landmarks:
            return frame
        annotated_frame = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_frame, landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                image=annotated_frame, landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        return annotated_frame

    def get_eye_features(self, results: Any) -> Optional[np.ndarray]:
        # 兼容旧的方法，供基础版使用
        if not results.multi_face_landmarks: return None
        face_landmarks = results.multi_face_landmarks[0]
        if len(face_landmarks.landmark) <= 473: return None
        l_iris, r_iris = face_landmarks.landmark[468], face_landmarks.landmark[473]
        return np.array([l_iris.x, l_iris.y, r_iris.x, r_iris.y])

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()