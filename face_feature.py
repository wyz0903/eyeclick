"""
面部特征点检测模块 (Face Landmark Detection Module)

该模块封装了 Google MediaPipe 的 Face Mesh 功能，作为一个独立的组件提供给主项目使用。
具备高内聚、低耦合的特点，支持提取面部特征点坐标以及在图像上进行可视化绘制。

依赖项:
    - opencv-python (cv2)
    - mediapipe
    - numpy
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Any


class FaceLandmarkDetector:
    """
    面部特征点检测器类。
    封装了 MediaPipe Face Mesh 模型，提供面部特征点提取和绘制功能。
    """

    def __init__(
            self,
            static_image_mode: bool = False,
            max_num_faces: int = 1,
            refine_landmarks: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        """
        初始化面部特征点检测器。

        参数:
            static_image_mode (bool): 如果为 True，则将输入视为一批静态可能不相关的图像。
                                      对于视频流，请设置为 False 以提升性能。默认为 False。
            max_num_faces (int): 允许检测的最大人脸数量。默认为 1。
            refine_landmarks (bool): 是否进一步细化眼睛周围、嘴唇和瞳孔的特征点。默认为 True。
            min_detection_confidence (float): 人脸检测模型的最小置信度阈值 (0.0 - 1.0)。默认为 0.5。
            min_tracking_confidence (float): 人脸追踪模型的最小置信度阈值 (0.0 - 1.0)。默认为 0.5。
        """
        # 初始化 MediaPipe 的 Face Mesh 模块
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 初始化 MediaPipe 的绘制工具，用于可视化
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_landmarks(self, frame: np.ndarray) -> Optional[Any]:
        """
        从输入图像中检测面部特征点。

        参数:
            frame (np.ndarray): 输入的图像帧 (BGR 格式，OpenCV 默认格式)。

        返回:
            Optional[Any]: 包含检测结果的 MediaPipe 对象。如果没有检测到人脸则返回 None。
                           结果对象中包含 multi_face_landmarks 属性。
        """
        # MediaPipe 需要 RGB 格式的图像，而 OpenCV 默认读取的是 BGR 格式
        # 为了提高性能，可以选择将图像标记为不可写，通过引用传递给模型
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理图像并提取特征点
        results = self.face_mesh.process(frame_rgb)

        # 恢复图像的可写属性，方便后续可能的操作（如绘制）
        frame.flags.writeable = True

        return results

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        在图像上绘制检测到的面部特征点网格。

        参数:
            frame (np.ndarray): 需要绘制的原始图像 (BGR 格式)。
            results (Any): `detect_landmarks` 方法返回的 MediaPipe 结果对象。

        返回:
            np.ndarray: 绘制了面部特征点网格后的新图像。
        """
        annotated_frame = frame.copy()

        if not results or not results.multi_face_landmarks:
            return annotated_frame

        # 遍历检测到的每一张人脸
        for face_landmarks in results.multi_face_landmarks:
            # 1. 绘制面部网格（Tesselation）
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # 2. 绘制面部轮廓（如眼睛、嘴唇、眉毛轮廓）
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # 3. 绘制瞳孔（如果 refine_landmarks 为 True 时生效）
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

        return annotated_frame

    def extract_landmark_coordinates(self, frame: np.ndarray, results: Any) -> List[List[Tuple[int, int]]]:
        """
        提取面部特征点的 2D 像素坐标，方便主项目进行后续的数据分析或逻辑处理。

        参数:
            frame (np.ndarray): 当前处理的图像，用于获取图像宽高。
            results (Any): MediaPipe 结果对象。

        返回:
            List[List[Tuple[int, int]]]: 返回一个列表，其中每个元素代表一张人脸，
                                         它本身也是一个列表，包含 (x, y) 像素坐标的元组。
        """
        faces_coordinates = []
        if not results or not results.multi_face_landmarks:
            return faces_coordinates

        image_height, image_width, _ = frame.shape

        for face_landmarks in results.multi_face_landmarks:
            face_coords = []
            for landmark in face_landmarks.landmark:
                # 将归一化的坐标 (0.0 - 1.0) 转换为实际图像像素坐标
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                face_coords.append((x, y))
            faces_coordinates.append(face_coords)

        return faces_coordinates

    def release(self):
        """
        释放 MediaPipe 资源。在不再需要检测器或程序退出时调用。
        """
        self.face_mesh.close()