"""
视线追踪数据集采集脚本 (Gaze Dataset Collector)

该脚本会在全屏显示一个随机游走的白点。
用户需盯着白点看，脚本会以固定的频率将“屏幕目标坐标”和“面部/眼部特征”记录到 CSV 文件中。

配置说明:
    - 修改 DOT_SPEED 改变白点移动速度。
    - 修改 SAMPLE_RATE 改变数据采样频率。
    - 修改 OUTPUT_FILE 更改导出的文件名。
"""

import cv2
import numpy as np
import time
import csv
import random
import math
from face_feature_MediaPipe import FaceLandmarkDetector

# ================= 核心配置区域 =================
DOT_SPEED = 600  # 白点移动速度 (像素/秒)
SAMPLE_RATE = 15  # 采样频率 (Hz，每秒记录几次数据)
OUTPUT_FILE = "gaze_dataset.csv"  # 导出的数据集文件名
DOT_RADIUS = 15  # 白点半径


# ===============================================

def get_screen_resolution():
    """尝试跨平台获取真实屏幕分辨率"""
    try:
        import tkinter as tk
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        print("无法自动获取屏幕分辨率，回退到默认值 1920x1080")
        return 1920, 1080


def main():
    SCREEN_W, SCREEN_H = get_screen_resolution()
    print(f"检测到屏幕分辨率: {SCREEN_W}x{SCREEN_H}")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    # 初始化全屏窗口
    window_name = "Dataset Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 白点位置初始化
    current_x, current_y = SCREEN_W / 2, SCREEN_H / 2
    target_x, target_y = current_x, current_y

    # 打开 CSV 文件并写入表头
    csv_file = open(OUTPUT_FILE, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    header = [
        "timestamp", "target_x", "target_y",
        "head_pitch", "head_yaw", "head_roll",
        "head_tx", "head_ty", "head_tz",
        "l_iris_rel_x", "l_iris_rel_y",
        "r_iris_rel_x", "r_iris_rel_y"
    ]
    # 动态添加 478 个面部特征点的 x, y, z 坐标列头 (共计 1434 列)
    for i in range(478):
        header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

    csv_writer.writerow(header)

    print(f"\n--- 数据集采集已准备就绪 ---")
    print(f"即将开始记录到: {OUTPUT_FILE}")
    print(f"每行将记录 {len(header)} 个数据维度。")
    print("请紧盯屏幕上的白点。按 'ESC' 键退出并保存。\n")

    # 时间控制变量
    prev_frame_time = time.time()
    last_sample_time = 0
    record_count = 0

    with FaceLandmarkDetector(refine_landmarks=True) as detector:
        while True:
            current_time = time.time()
            dt = current_time - prev_frame_time
            prev_frame_time = current_time

            # 1. 读取并处理摄像头画面 (保持连续处理以维持特征点的平滑追踪)
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h_img, w_img = frame.shape[:2]
            results = detector.process_frame(frame)

            # 2. 更新白点平滑移动逻辑
            dist = math.hypot(target_x - current_x, target_y - current_y)
            if dist < DOT_SPEED * dt:
                # 已到达目标点，生成下一个随机目标点 (留出边缘 margin 以防点飞出屏幕)
                current_x, current_y = target_x, target_y
                margin = 50
                target_x = random.randint(margin, SCREEN_W - margin)
                target_y = random.randint(margin, SCREEN_H - margin)
            else:
                # 沿向量向目标点移动
                dx = (target_x - current_x) / dist
                dy = (target_y - current_y) / dist
                current_x += dx * DOT_SPEED * dt
                current_y += dy * DOT_SPEED * dt

            # 3. 采样与记录数据
            if current_time - last_sample_time >= 1.0 / SAMPLE_RATE:
                features = detector.extract_comprehensive_features(results, (h_img, w_img))
                if features and results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # 将基础特征写入行列表
                    row = [
                        f"{current_time:.4f}",
                        int(current_x), int(current_y),
                        f"{features['head_pitch']:.4f}", f"{features['head_yaw']:.4f}", f"{features['head_roll']:.4f}",
                        f"{features['head_tx']:.4f}", f"{features['head_ty']:.4f}", f"{features['head_tz']:.4f}",
                        f"{features['l_iris_rel_x']:.4f}", f"{features['l_iris_rel_y']:.4f}",
                        f"{features['r_iris_rel_x']:.4f}", f"{features['r_iris_rel_y']:.4f}"
                    ]

                    # 遍历并追加所有 478 个原始 3D 特征点坐标
                    # 这里的 x, y 是 [0.0, 1.0] 的归一化坐标，z 是相对深度比例
                    for lm in landmarks:
                        row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])

                    csv_writer.writerow(row)
                    last_sample_time = current_time
                    record_count += 1

            # 4. 绘制屏幕UI
            # 创建全黑纯色背景
            screen_display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

            # 绘制游走白点
            cv2.circle(screen_display, (int(current_x), int(current_y)), DOT_RADIUS, (255, 255, 255), -1)

            # 在左上角显示不显眼的状态文本
            cv2.putText(screen_display, f"Recording... Collected: {record_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(screen_display, "Press ESC to stop", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv2.imshow(window_name, screen_display)

            if cv2.waitKey(1) & 0xFF == 27:  # 27 是 ESC 键的 ASCII 码
                break

    # 释放资源与保存文件
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"数据采集结束。共采集 {record_count} 条数据，已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()