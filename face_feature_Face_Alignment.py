import cv2
import numpy as np
import face_alignment
from typing import List, Optional, Tuple, Union


class FaceLandmarkDetector:
    """
    面部特征点检测器组件。

    该类封装了基于 face_alignment 库的面部特征点提取功能。
    设计为大型项目中的一个可插拔组件，支持图像格式转换、推理以及结果的可视化。
    """

    def __init__(self,
                 enable_3d: bool = True,
                 device: str = 'cpu',
                 face_detector: str = 'sfd'):
        """
        初始化面部特征点检测器。

        Args:
            enable_3d (bool): 是否提取 3D 特征点。默认为 False (提取 2D 特征点)。
            device (str): 运行设备，'cpu' 或 'cuda'。强烈建议在有 GPU 的设备上使用 'cuda' 以保证实时性。
            face_detector (str): 底层使用的人脸检测器，可选 'sfd', 'blazeface' 等。
        """
        self.enable_3d = enable_3d
        self.device = device

        # 动态选择特征点类型 (2D 或 3D)，以兼容不同版本的 face-alignment 库
        if hasattr(face_alignment.LandmarksType, '_2D'):
            landmarks_type = face_alignment.LandmarksType._3D if enable_3d else face_alignment.LandmarksType._2D
        elif hasattr(face_alignment.LandmarksType, 'TWO_D'):
            landmarks_type = face_alignment.LandmarksType.THREE_D if enable_3d else face_alignment.LandmarksType.TWO_D
        else:
            # 兜底策略：强制按枚举列表索引获取
            enum_list = list(face_alignment.LandmarksType)
            landmarks_type = enum_list[-1] if enable_3d else enum_list[0]

        print(
            f"[INFO] 正在初始化 FaceAlignment 模型 (设备: {device}, 3D: {enable_3d})... 这可能需要一些时间下载模型权重。")

        try:
            self.model = face_alignment.FaceAlignment(
                landmarks_type,
                flip_input=False,
                device=device,
                face_detector=face_detector
            )
            print("[INFO] FaceAlignment 模型初始化成功！")
        except Exception as e:
            print(f"[ERROR] FaceAlignment 初始化失败: {e}")
            raise e

    def detect_landmarks(self, image_bgr: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        从 BGR 图像中检测面部特征点。

        Args:
            image_bgr (np.ndarray): OpenCV 读取的 BGR 格式图像。

        Returns:
            Optional[List[np.ndarray]]: 返回包含多个面部特征点数组的列表。
                                        如果没有检测到人脸，则返回 None。
                                        每个 np.ndarray 的形状通常为 (68, 2) 或 (68, 3)。
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        # face_alignment 库要求输入图像为 RGB 格式
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 进行推理，获取特征点
        landmarks = self.model.get_landmarks(image_rgb)

        return landmarks

    def draw_landmarks(self,
                       image_bgr: np.ndarray,
                       landmarks_list: Optional[List[np.ndarray]],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 2) -> np.ndarray:
        """
        将检测到的特征点绘制到图像上。

        Args:
            image_bgr (np.ndarray): 原始 BGR 图像。
            landmarks_list (Optional[List[np.ndarray]]): detect_landmarks 返回的特征点列表。
            color (Tuple[int, int, int]): 特征点的颜色 (B, G, R)。默认为绿色。
            radius (int): 绘制的圆点半径。

        Returns:
            np.ndarray: 绘制了特征点的新图像。
        """
        # 为了不破坏原始图像，创建副本
        output_image = image_bgr.copy()

        if landmarks_list is None:
            return output_image

        for face_landmarks in landmarks_list:
            # face_landmarks 是一个形状为 (68, 2) 或 (68, 3) 的 numpy 数组
            for pt in face_landmarks:
                # 提取 x 和 y 坐标并转换为整数
                x, y = int(pt[0]), int(pt[1])
                # 在图像上绘制实心圆
                cv2.circle(output_image, (x, y), radius, color, -1)

        return output_image

    def process_and_draw(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        高级封装 API：输入图像，直接返回处理后的图像和特征点数据。
        方便在大型流水线(Pipeline)中直接调用。

        Args:
            image_bgr (np.ndarray): 输入的 BGR 图像。

        Returns:
            Tuple[np.ndarray, Optional[List[np.ndarray]]]: (绘制好的图像, 特征点列表)
        """
        landmarks = self.detect_landmarks(image_bgr)
        drawn_image = self.draw_landmarks(image_bgr, landmarks)
        return drawn_image, landmarks