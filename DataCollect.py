"""
视线追踪高级数据集采集脚本 (Gaze Dataset Collector Pro)

- 采用 HDF5 格式存储，实现图像与一维特征的高效打包。
- 按精细网格顺序生成凝视点，支持断点续录。
- 自动裁剪、保存双眼区域的高清小图，为后续多模态 CNN 训练准备素材。
- 要求用户在白点收集期间，盯着点看的同时充分移动头部（转动、俯仰、靠近远离）。

依赖项: pip install h5py opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import time
import h5py
import os
from face_feature_MediaPipe import FaceLandmarkDetector

# ================= 核心配置区域 =================
SAMPLE_RATE = 15             # 采样频率 (Hz)
OUTPUT_FILE = "gaze_dataset.h5"  # 采用高效的 HDF5 格式替代 CSV
DOT_RADIUS = 5               # 凝视点半径

GRID_COLS = 120               # 网格列数
GRID_ROWS = 80              # 网格行数
DELAY_BEFORE_COLLECT = 1.0   # 准备期（黄点）：寻找点并准备，时长1秒
COLLECT_DURATION = 40.0       # 采集期（白点）：紧盯点并进行头部运动，时长5秒

EYE_IMG_SIZE = (64, 36)      # 裁剪出的双眼图像尺寸 (宽, 高)
EYE_CROP_PADDING = 15        # 裁剪边界的向外扩充像素
# ===============================================

# 左右眼的 MediaPipe 关键点索引 (用于框选眼部区域)
# 这里的"左眼"指的是画面左侧的眼 (在做了 cv2.flip 之后也就是用户的左脸眼)
LEFT_EYE_INDICES =  [33 ,   7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

class HDF5DatasetWriter:
    """管理 HDF5 文件的初始化、缓冲与写入逻辑"""
    def __init__(self, filename, buffer_size=30):
        self.filename = filename
        self.buffer_size = buffer_size
        self.reset_buffers()

        # 初始化或打开 HDF5 文件
        self.file = h5py.File(filename, 'a')

        # 预创建动态扩容的数据集
        if 'eye_l' not in self.file:
            print("初始化新的 HDF5 数据集结构...")
            self.file.create_dataset('eye_l', shape=(0, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 3),
                                     maxshape=(None, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 3), dtype='uint8', chunks=True)
            self.file.create_dataset('eye_r', shape=(0, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 3),
                                     maxshape=(None, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 3), dtype='uint8', chunks=True)
            self.file.create_dataset('features', shape=(0, 1444), maxshape=(None, 1444), dtype='float32', chunks=True)
            self.file.create_dataset('targets', shape=(0, 2), maxshape=(None, 2), dtype='float32', chunks=True)

    def reset_buffers(self):
        self.buffer_eye_l = []
        self.buffer_eye_r = []
        self.buffer_features = []
        self.buffer_targets = []

    def get_total_samples(self):
        return self.file['targets'].shape[0] + len(self.buffer_targets)

    def add_sample(self, eye_l, eye_r, features, target):
        self.buffer_eye_l.append(eye_l)
        self.buffer_eye_r.append(eye_r)
        self.buffer_features.append(features)
        self.buffer_targets.append(target)

        if len(self.buffer_targets) >= self.buffer_size:
            self.flush()

    def flush(self):
        if len(self.buffer_targets) == 0: return

        n_new = len(self.buffer_targets)
        n_old = self.file['targets'].shape[0]
        n_total = n_old + n_new

        # 动态扩大 HDF5 容量
        self.file['eye_l'].resize(n_total, axis=0)
        self.file['eye_r'].resize(n_total, axis=0)
        self.file['features'].resize(n_total, axis=0)
        self.file['targets'].resize(n_total, axis=0)

        # 写入数据
        self.file['eye_l'][n_old:n_total] = np.array(self.buffer_eye_l)
        self.file['eye_r'][n_old:n_total] = np.array(self.buffer_eye_r)
        self.file['features'][n_old:n_total] = np.array(self.buffer_features)
        self.file['targets'][n_old:n_total] = np.array(self.buffer_targets)

        self.reset_buffers()

    def close(self):
        self.flush()
        self.file.close()

def extract_eye_image(frame, landmarks, eye_indices, w, h):
    """根据关键点从原始帧中裁剪并缩放眼部区域"""
    xs = [int(landmarks[idx].x * w) for idx in eye_indices]
    ys = [int(landmarks[idx].y * h) for idx in eye_indices]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 增加边界 Padding 并防止越界
    x_min = max(0, x_min - EYE_CROP_PADDING)
    y_min = max(0, y_min - EYE_CROP_PADDING)
    x_max = min(w, x_max + EYE_CROP_PADDING)
    y_max = min(h, y_max + EYE_CROP_PADDING)

    eye_img = frame[y_min:y_max, x_min:x_max]

    # 安全检查并重设大小
    if eye_img.size == 0:
        return np.zeros((EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 3), dtype=np.uint8)
    return cv2.resize(eye_img, EYE_IMG_SIZE)

def get_screen_resolution():
    try:
        import tkinter as tk
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080

def build_grid_points(screen_w, screen_h):
    """根据屏幕大小和配置生成网格点序列"""
    margin_x = int(screen_w * 0.05)
    margin_y = int(screen_h * 0.05)
    xs = np.linspace(margin_x, screen_w - margin_x, GRID_COLS)
    ys = np.linspace(margin_y, screen_h - margin_y, GRID_ROWS)
    return [(int(x), int(y)) for y in ys for x in xs]

def main():
    SCREEN_W, SCREEN_H = get_screen_resolution()
    grid_points = build_grid_points(SCREEN_W, SCREEN_H)
    samples_per_point = int(SAMPLE_RATE * COLLECT_DURATION)

    # 初始化摄像头和数据集写入器
    cap = cv2.VideoCapture(0)
    writer = HDF5DatasetWriter(OUTPUT_FILE)

    # 智能推算进度 (断点续传逻辑)
    total_samples = writer.get_total_samples()
    start_grid_idx = total_samples // samples_per_point

    print(f"\n===== 视线多模态采集系统 =====")
    print(f"屏幕分辨率: {SCREEN_W}x{SCREEN_H} | 网格数量: {len(grid_points)}")
    print(f"已有数据量: {total_samples} 帧 | 将从第 {start_grid_idx + 1} 个网格点继续")
    print(f"==============================\n")
    print("【操作说明】")
    print(" 🟡 黄点出现：请移动视线找到该点，眼球对准。")
    print(" ⚪ 白点出现：开始记录！紧盯白点，同时让你的头部充分运动（左右摇头、上下点头、靠近后仰）。")
    print(" 请按 'ESC' 键退出，所有数据会自动安全保存。\n")

    window_name = "Dataset Collector Pro"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_grid_idx = start_grid_idx
    point_spawn_time = time.time()
    last_sample_time = 0
    session_records = 0

    try:
        with FaceLandmarkDetector(refine_landmarks=True) as detector:
            while True:
                # 检查是否完成了一轮网格采集
                if current_grid_idx >= len(grid_points):
                    print("恭喜！所有网格点已采集完毕。重新开始新一轮的网格覆盖...")
                    current_grid_idx = 0

                target_x, target_y = grid_points[current_grid_idx]
                current_time = time.time()
                time_since_spawn = current_time - point_spawn_time

                # 1. 读取摄像头
                success, frame = cap.read()
                if not success: continue
                frame = cv2.flip(frame, 1)
                h_img, w_img = frame.shape[:2]

                results = detector.process_frame(frame)

                # 2. 状态机
                if time_since_spawn < DELAY_BEFORE_COLLECT:
                    is_collecting = False
                    current_radius = DOT_RADIUS + 4
                    dot_color = (0, 255, 255) # BGR 黄点准备
                    status_text = "READY..."
                elif time_since_spawn < DELAY_BEFORE_COLLECT + COLLECT_DURATION:
                    is_collecting = True
                    current_radius = DOT_RADIUS
                    dot_color = (255, 255, 255) # BGR 白点采集
                    status_text = "FOCUS & MOVE HEAD!"
                else:
                    current_grid_idx += 1
                    point_spawn_time = time.time()
                    continue

                # 3. 提取特征、图像并采样记录
                if is_collecting and (current_time - last_sample_time >= 1.0 / SAMPLE_RATE):
                    features_dict = detector.extract_comprehensive_features(results, (h_img, w_img))
                    if features_dict and results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark

                        # (A) 提取眼部图像
                        eye_l_img = extract_eye_image(frame, landmarks, LEFT_EYE_INDICES, w_img, h_img)
                        eye_r_img = extract_eye_image(frame, landmarks, RIGHT_EYE_INDICES, w_img, h_img)

                        # (B) 组装 1444 维特征向量 (10个高级特征 + 478*3个坐标)
                        vec_features = [
                            features_dict['head_pitch'], features_dict['head_yaw'], features_dict['head_roll'],
                            features_dict['head_tx'], features_dict['head_ty'], features_dict['head_tz'],
                            features_dict['l_iris_rel_x'], features_dict['l_iris_rel_y'],
                            features_dict['r_iris_rel_x'], features_dict['r_iris_rel_y']
                        ]
                        for lm in landmarks:
                            vec_features.extend([lm.x, lm.y, lm.z])

                        # (C) 屏幕注视坐标
                        target = [target_x, target_y]

                        # 写入缓冲
                        writer.add_sample(eye_l_img, eye_r_img, vec_features, target)

                        last_sample_time = current_time
                        session_records += 1

                # 4. 绘制 UI
                screen_display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                cv2.circle(screen_display, (target_x, target_y), current_radius, dot_color, -1)

                # 顶部信息栏
                cv2.putText(screen_display, f"Point {current_grid_idx+1}/{len(grid_points)} | Status: {status_text}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_collecting else (0, 255, 255), 2)
                cv2.putText(screen_display, f"Session Saved: {session_records} frames | Press ESC to stop",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                cv2.imshow(window_name, screen_display)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        # 无论正常退出还是崩溃，强制刷新并关闭文件以保护 HDF5 不被损坏
        writer.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n数据流已安全保存。本次运行时长采集了 {session_records} 帧。")

if __name__ == "__main__":
    main()