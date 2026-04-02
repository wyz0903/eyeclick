"""
视线追踪数据集采集脚本 - 定点凝视改进版 (Gaze Dataset Collector)

该脚本会在屏幕上随机生成固定位置的凝视点。
- 点出现的前 0.5 秒为【准备期】，提示为黄色，不记录数据（供眼球稳定）。
- 0.5 秒后转为【采集期】，变为较小的白点，以特定频率记录特征数据。
- 支持断点续录：再次运行脚本会自动向已有文件追加数据，方便分批次采集。

配置说明:
    - 修改 SAMPLE_RATE 改变数据采样频率。
    - 修改 DELAY_BEFORE_COLLECT 改变稳定时间。
    - 修改 COLLECT_DURATION 改变每个点的采集时长。
"""

import cv2
import numpy as np
import time
import csv
import random
import os
from face_feature_MediaPipe import FaceLandmarkDetector

# ================= 核心配置区域 =================
SAMPLE_RATE = 15             # 采样频率 (Hz，每秒记录几次数据)
OUTPUT_FILE = "gaze_dataset.csv"  # 导出的数据集文件名
DOT_RADIUS = 6               # 白点半径（设小一点以提高凝视精度）
DELAY_BEFORE_COLLECT = 0.5   # 点出现后，等待多少秒再开始收集数据（防眼动误差）
COLLECT_DURATION = 2.0       # 每次有效收集数据的持续时间（秒）
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

    # 准备表头
    header = [
        "timestamp", "target_x", "target_y",
        "head_pitch", "head_yaw", "head_roll",
        "head_tx", "head_ty", "head_tz",
        "l_iris_rel_x", "l_iris_rel_y",
        "r_iris_rel_x", "r_iris_rel_y"
    ]
    for i in range(478):
        header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

    # 检查是否已存在数据集，如果存在则进入追加模式
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, mode='a', newline='') # 使用 'a' 追加模式
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        csv_writer.writerow(header)
        print(f"\n--- 创建新数据集: {OUTPUT_FILE} ---")
    else:
        print(f"\n--- 发现已有数据集，将向 {OUTPUT_FILE} 追加数据 ---")

    print(f"每行将记录 {len(header)} 个数据维度。")
    print("规则: 黄点出现时寻找目标，白点出现时紧盯白点。")
    print("请按 'ESC' 键退出并保存。\n")

    # 初始点状态变量
    margin = 50
    target_x = random.randint(margin, SCREEN_W - margin)
    target_y = random.randint(margin, SCREEN_H - margin)
    point_spawn_time = time.time()
    last_sample_time = 0
    record_count = 0 # 本次运行新增记录数

    with FaceLandmarkDetector(refine_landmarks=True) as detector:
        while True:
            current_time = time.time()
            time_since_spawn = current_time - point_spawn_time

            # 1. 读取摄像头
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h_img, w_img = frame.shape[:2]
            results = detector.process_frame(frame)

            # 2. 状态机逻辑判定
            if time_since_spawn < DELAY_BEFORE_COLLECT:
                # 准备期：不收集数据，使用稍大一点的黄点提示用户位置
                is_collecting = False
                current_radius = DOT_RADIUS + 4
                dot_color = (0, 255, 255) # BGR 黄色
            elif time_since_spawn < DELAY_BEFORE_COLLECT + COLLECT_DURATION:
                # 收集期：收集数据，使用精确的白点
                is_collecting = True
                current_radius = DOT_RADIUS
                dot_color = (255, 255, 255) # BGR 白色
            else:
                # 时间到，刷新到下一个点，并重新计时
                target_x = random.randint(margin, SCREEN_W - margin)
                target_y = random.randint(margin, SCREEN_H - margin)
                point_spawn_time = time.time()
                continue # 重新开始下一次循环

            # 3. 采样与记录数据
            if is_collecting and (current_time - last_sample_time >= 1.0 / SAMPLE_RATE):
                features = detector.extract_comprehensive_features(results, (h_img, w_img))
                if features and results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    row = [
                        f"{current_time:.4f}",
                        int(target_x), int(target_y),
                        f"{features['head_pitch']:.4f}", f"{features['head_yaw']:.4f}", f"{features['head_roll']:.4f}",
                        f"{features['head_tx']:.4f}", f"{features['head_ty']:.4f}", f"{features['head_tz']:.4f}",
                        f"{features['l_iris_rel_x']:.4f}", f"{features['l_iris_rel_y']:.4f}",
                        f"{features['r_iris_rel_x']:.4f}", f"{features['r_iris_rel_y']:.4f}"
                    ]

                    for lm in landmarks:
                        row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])

                    csv_writer.writerow(row)
                    last_sample_time = current_time
                    record_count += 1

            # 4. 绘制屏幕UI
            screen_display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

            # 绘制标定点
            cv2.circle(screen_display, (target_x, target_y), current_radius, dot_color, -1)

            # 状态提示文本
            status_text = "FOCUS..." if is_collecting else "GET READY"
            cv2.putText(screen_display, f"Status: {status_text} | Session Records: {record_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(screen_display, "Press ESC to save and pause", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv2.imshow(window_name, screen_display)

            if cv2.waitKey(1) & 0xFF == 27: # 27 是 ESC 键
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\n已安全停止！本次运行共采集了 {record_count} 条数据。")

if __name__ == "__main__":
    main()