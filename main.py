import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# --- 兼容性修复：显式导入以避免部分环境下的 AttributeError ---
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
except (ImportError, AttributeError):
    # 如果底层导入失败，则回退到默认常规方式
    mp_face_mesh_module = mp.solutions.face_mesh

# 开启 PyAutoGUI 防故障（将鼠标移到屏幕四个角可以强制中断，防止失控）
pyautogui.FAILSAFE = True


class FeatureExtractor:
    """
    视觉感知：提取眼眶边界和虹膜中心
    """

    def __init__(self):
        # 使用兼容性修复后的模块引用
        self.mp_face_mesh = mp_face_mesh_module
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # MediaPipe 眼部关键点索引
        # 左眼 (用户真实左眼)
        self.LEFT_EYE_LEFT = 33  # 外眼角
        self.LEFT_EYE_RIGHT = 133  # 内眼角
        self.LEFT_EYE_TOP = 159  # 上眼睑
        self.LEFT_EYE_BOTTOM = 145  # 下眼睑
        self.LEFT_IRIS = 468  # 虹膜中心

        # 右眼 (用户真实右眼)
        self.RIGHT_EYE_LEFT = 362  # 内眼角
        self.RIGHT_EYE_RIGHT = 263  # 外眼角
        self.RIGHT_EYE_TOP = 386  # 上眼睑
        self.RIGHT_EYE_BOTTOM = 374  # 下眼睑
        self.RIGHT_IRIS = 473  # 虹膜中心

    def get_distance(self, p1, p2):
        """计算两点间的欧几里得距离"""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        features = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 提取左眼特征
            l_left = landmarks[self.LEFT_EYE_LEFT]
            l_right = landmarks[self.LEFT_EYE_RIGHT]
            l_top = landmarks[self.LEFT_EYE_TOP]
            l_bottom = landmarks[self.LEFT_EYE_BOTTOM]
            l_iris = landmarks[self.LEFT_IRIS]

            # 提取右眼特征
            r_left = landmarks[self.RIGHT_EYE_LEFT]
            r_right = landmarks[self.RIGHT_EYE_RIGHT]
            r_top = landmarks[self.RIGHT_EYE_TOP]
            r_bottom = landmarks[self.RIGHT_EYE_BOTTOM]
            r_iris = landmarks[self.RIGHT_IRIS]

            # 计算眼睛张开度 (EAR - Eye Aspect Ratio) 用于检测眨眼
            l_ear = self.get_distance(l_top, l_bottom) / (self.get_distance(l_left, l_right) + 1e-6)
            r_ear = self.get_distance(r_top, r_bottom) / (self.get_distance(r_left, r_right) + 1e-6)
            avg_ear = (l_ear + r_ear) / 2.0

            # 计算虹膜在眼眶内的水平相对位置 (X轴比率)
            # 0.0 表示看向最左侧，1.0 表示看向最右侧
            l_ratio_x = (l_iris.x - l_left.x) / (l_right.x - l_left.x + 1e-6)
            r_ratio_x = (r_iris.x - r_left.x) / (r_right.x - r_left.x + 1e-6)
            avg_ratio_x = (l_ratio_x + r_ratio_x) / 2.0

            # 计算虹膜在眼眶内的垂直相对位置 (Y轴比率)
            l_ratio_y = (l_iris.y - l_top.y) / (l_bottom.y - l_top.y + 1e-6)
            r_ratio_y = (r_iris.y - r_top.y) / (r_bottom.y - r_top.y + 1e-6)
            avg_ratio_y = (l_ratio_y + r_ratio_y) / 2.0

            features = {
                'ratio_x': avg_ratio_x,
                'ratio_y': avg_ratio_y,
                'ear': avg_ear,
                'l_iris_pos': (l_iris.x, l_iris.y),  # 仅用于可视化
                'r_iris_pos': (r_iris.x, r_iris.y)  # 仅用于可视化
            }
        return features


class GazeMapper:
    """
    映射计算：将眼眶内的相对比例映射到屏幕坐标
    """

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 敏感度范围 (硬编码的经验值，不同人的眼球结构可能略有不同)
        # 正常人眼球转动时，虹膜中心在眼宽中的比例大约在 0.35 到 0.65 之间变动
        self.min_x_ratio = 0.35
        self.max_x_ratio = 0.65

        # 垂直方向同理
        self.min_y_ratio = 0.35
        self.max_y_ratio = 0.70

    def predict_screen_coords(self, features):
        rx = features['ratio_x']
        ry = features['ratio_y']

        # 将比例归一化到 0.0 - 1.0 之间
        norm_x = (rx - self.min_x_ratio) / (self.max_x_ratio - self.min_x_ratio)
        norm_y = (ry - self.min_y_ratio) / (self.max_y_ratio - self.min_y_ratio)

        # 钳制数值，防止越界
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # 因为后续我们在主循环中对图像进行了镜像翻转(flip)，
        # 所以这里的 X 轴坐标刚好是符合直觉的，不需要再做反转了
        screen_x = int(norm_x * self.screen_w)
        screen_y = int(norm_y * self.screen_h)

        return screen_x, screen_y


class MouseController:
    """
    鼠标控制：平滑移动与点击逻辑
    """

    def __init__(self):
        self.prev_x, self.prev_y = pyautogui.position()
        # 平滑系数：越小越平滑但延迟越大 (0.05 适合眼动追踪这种抖动较大的场景)
        self.smooth_factor = 0.1

        # 点击控制
        self.is_active = False  # 默认关闭鼠标控制，按 T 键开启
        self.blink_threshold = 0.18  # 闭眼阈值
        self.last_click_time = 0

    def move_and_click(self, target_x, target_y, ear):
        if not self.is_active:
            return

        # 平滑移动 (EMA 滤波)
        smooth_x = self.prev_x + (target_x - self.prev_x) * self.smooth_factor
        smooth_y = self.prev_y + (target_y - self.prev_y) * self.smooth_factor

        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y))
            self.prev_x, self.prev_y = smooth_x, smooth_y
        except pyautogui.FailSafeException:
            # 触发了防故障机制 (鼠标到了角落)
            self.is_active = False
            print("触发防故障机制，已自动暂停控制。")

        # 眨眼点击逻辑 (防抖：0.5秒内只能点击一次)
        current_time = time.time()
        if ear < self.blink_threshold and (current_time - self.last_click_time) > 0.5:
            pyautogui.click()
            self.last_click_time = current_time
            return True  # 发生点击
        return False


def main():
    screen_w, screen_h = pyautogui.size()
    print(f"检测到屏幕分辨率: {screen_w}x{screen_h}")

    extractor = FeatureExtractor()
    mapper = GazeMapper(screen_w, screen_h)
    mouse = MouseController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("========================================")
    print("系统启动成功！")
    print("【操作指南】")
    print(" 1. 按 't' 键：开启/暂停 鼠标控制 (Toggle)")
    print(" 2. 用力眨眼：触发鼠标左键点击")
    print(" 3. 按 'q' 键：退出程序")
    print("========================================")

    while True:
        success, frame = cap.read()  # 修复了之前的 cap.cap() 错误
        if not success:
            continue

        # 镜像翻转，符合照镜子的直觉
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 提取特征
        features = extractor.process_frame(frame)

        # 状态显示文本
        status_text = "ACTIVE" if mouse.is_active else "PAUSED (Press 'T')"
        color = (0, 255, 0) if mouse.is_active else (0, 0, 255)
        cv2.putText(frame, f"State: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if features:
            # 预测屏幕坐标
            screen_x, screen_y = mapper.predict_screen_coords(features)

            # 控制鼠标与检测眨眼
            clicked = mouse.move_and_click(screen_x, screen_y, features['ear'])

            # --- 可视化反馈 ---
            # 画出虹膜位置
            lx, ly = int(features['l_iris_pos'][0] * w), int(features['l_iris_pos'][1] * h)
            rx, ry = int(features['r_iris_pos'][0] * w), int(features['r_iris_pos'][1] * h)
            cv2.circle(frame, (lx, ly), 4, (255, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 4, (255, 255, 0), -1)

            # 显示调试信息
            cv2.putText(frame, f"Screen: {screen_x}, {screen_y}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 200, 0), 2)
            cv2.putText(frame, f"EAR (Blink): {features['ear']:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 200, 0), 2)

            if clicked:
                cv2.putText(frame, "CLICKED!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 显示画面
        cv2.imshow('EyeClick MVP Tracker', frame)

        # 键盘事件监听
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            mouse.is_active = not mouse.is_active  # 切换激活状态

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()