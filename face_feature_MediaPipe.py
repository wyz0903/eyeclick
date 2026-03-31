"""
面部特征点检测组件 (Face Landmark Detector)

该模块封装了 Google MediaPipe 的 Face Mesh 模型。
作为大型项目的一个子组件，它负责处理输入的图像帧，
提取 478 个面部特征点（包含虹膜等细节），并提供可视化辅助方法。

依赖项:
    - opencv-python (cv2)
    - mediapipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Any


class FaceLandmarkDetector:
    """
    面部特征点检测器类。

    封装了 MediaPipe Face Mesh 方案，提供图像处理、特征点提取和绘制功能。
    支持作为上下文管理器 (Context Manager) 使用，以确保资源正确释放。
    """

    def __init__(
            self,
            static_image_mode: bool = False,
            max_num_faces: int = 1,
            refine_landmarks: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
    ):
        """
        初始化面部特征点检测器。

        参数:
            static_image_mode: 如果设为 True，则将输入图像视为独立的静态图片；
                               如果设为 False，则将其视为视频流，以提高跟踪效率。
            max_num_faces: 允许检测的最大面部数量。
            refine_landmarks: 是否进一步细化眼睛、嘴唇周围的特征点，并输出虹膜特征点（总共478个点）。
            min_detection_confidence: 初始检测的最小置信度阈值 (0.0 ~ 1.0)。
            min_tracking_confidence: 跟踪的最小置信度阈值 (0.0 ~ 1.0)。
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化 MediaPipe FaceMesh 对象
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Any:
        """
        处理单帧图像，提取面部特征点。

        参数:
            frame: OpenCV 格式的 BGR 图像帧 (numpy array)。

        返回:
            results: MediaPipe 处理后的结果对象。包含 .multi_face_landmarks 属性。
        """
        # 为了提高性能，可以选择将图像标记为不可写，以通过引用传递
        frame.flags.writeable = False

        # OpenCV 默认使用 BGR 颜色空间，而 MediaPipe 需要 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理图像并返回结果
        results = self.face_mesh.process(frame_rgb)

        # 恢复图像可写状态
        frame.flags.writeable = True

        return results

    def extract_landmarks(
            self,
            results: Any,
            image_shape: Tuple[int, int, int]
    ) -> List[List[Tuple[int, int]]]:
        """
        从处理结果中提取像素坐标格式的特征点数据，方便下游任务处理。

        参数:
            results: process_frame() 返回的 MediaPipe 结果对象。
            image_shape: 输入图像的形状 (height, width, channels)。

        返回:
            一个列表，包含检测到的所有面部。每个面部是一个包含 (x, y) 像素坐标元组的列表。
            如果没有检测到人脸，则返回空列表。
        """
        faces_landmarks = []
        if not results.multi_face_landmarks:
            return faces_landmarks

        h, w, _ = image_shape
        for face_landmarks in results.multi_face_landmarks:
            # 将归一化坐标转换为像素坐标
            pixel_landmarks = [
                (int(landmark.x * w), int(landmark.y * h))
                for landmark in face_landmarks.landmark
            ]
            faces_landmarks.append(pixel_landmarks)

        return faces_landmarks

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        在给定的图像帧上绘制检测到的面部特征点及网格。

        参数:
            frame: 原始 OpenCV BGR 图像帧。
            results: process_frame() 返回的 MediaPipe 结果对象。

        返回:
            绘制了特征点的 BGR 图像帧。
        """
        if not results.multi_face_landmarks:
            return frame

        annotated_frame = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            # 绘制面部网格底图
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # 绘制面部轮廓
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # 绘制虹膜 (如果 refine_landmarks 为 True)
            if hasattr(self.mp_face_mesh, 'FACEMESH_IRISES'):
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        return annotated_frame

    def close(self):
        """释放 MediaPipe 资源。"""
        self.face_mesh.close()

    def __enter__(self):
        """支持 'with' 语句的上下文管理器进入方法。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 'with' 语句的上下文管理器退出方法，自动释放资源。"""
        self.close()