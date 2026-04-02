"""
视线预测实时测试脚本 (Gaze Inference Test)

调用已训练的 GazeResNet 模型，通过前置摄像头实时读取 478 个特征点，
预测屏幕注视点坐标，并在全屏黑色背景上绘制一个白点。

依赖项:
    - face_landmark_detector.py (特征提取组件)
    - train_gaze_model.py (网络结构和配置)
"""

import cv2
import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from face_feature_MediaPipe import FaceLandmarkDetector
from train import GazeResNet, CONFIG


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


def get_fitted_scaler(csv_path):
    """
    为了将模型的输出还原为真实的屏幕像素坐标，
    我们需要重新拟合训练时使用的 StandardScaler。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据集文件 {csv_path}。需要它来还原坐标比例。")
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    scaler.fit(df[['target_x', 'target_y']].values)
    return scaler


def main():
    SCREEN_W, SCREEN_H = get_screen_resolution()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")

    # 1. 初始化坐标缩放器
    print("正在加载坐标缩放器...")
    scaler_y = get_fitted_scaler(CONFIG['dataset_path'])

    # 2. 初始化模型并加载权重
    print("正在加载模型权重...")
    model = GazeResNet().to(device)
    if not os.path.exists(CONFIG['model_save_path']):
        raise FileNotFoundError(f"找不到模型权重文件: {CONFIG['model_save_path']}。请先运行训练脚本。")
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device, weights_only=True))
    model.eval()  # 设置为推理模式

    # 3. 初始化摄像头和全屏窗口
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    window_name = "Gaze Prediction Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--- 实时预测已启动 ---")
    print("屏幕上将显示白点作为预测的视线落点。")
    print("按 'ESC' 键退出。\n")

    # 平滑滤波器变量 (可选，用于减少白点抖动)
    smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2
    alpha = 0.3  # 平滑系数，越小越平滑但延迟越高

    with FaceLandmarkDetector(refine_landmarks=True) as detector:
        with torch.no_grad():  # 推理时不需要计算梯度
            while True:
                success, frame = cap.read()
                if not success: continue

                frame = cv2.flip(frame, 1)
                results = detector.process_frame(frame)

                # 创建全黑屏幕背景
                screen_display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # 提取 478 个点的 (x, y, z)
                    raw_features = []
                    for lm in landmarks:
                        raw_features.append([lm.x, lm.y, lm.z])

                    # 转换为 Numpy 数组并重塑维度以匹配模型输入: (1, 3, 478)
                    X_np = np.array(raw_features, dtype=np.float32)  # 形状: (478, 3)
                    X_np = X_np.transpose(1, 0)  # 形状: (3, 478)
                    X_np = np.expand_dims(X_np, axis=0)  # 形状: (1, 3, 478)

                    # 转为 Tensor 并送入设备计算
                    X_tensor = torch.tensor(X_np).to(device)
                    pred_scaled = model(X_tensor).cpu().numpy()

                    # 使用 scaler 将归一化的坐标逆转换回真实的屏幕像素坐标
                    pred_pixel = scaler_y.inverse_transform(pred_scaled)[0]
                    target_x, target_y = pred_pixel[0], pred_pixel[1]

                    # 应用指数移动平均 (EMA) 平滑白点移动，防止剧烈抖动
                    smooth_x = alpha * target_x + (1 - alpha) * smooth_x
                    smooth_y = alpha * target_y + (1 - alpha) * smooth_y

                    # 限制在屏幕范围内
                    draw_x = int(max(0, min(SCREEN_W, smooth_x)))
                    draw_y = int(max(0, min(SCREEN_H, smooth_y)))

                    # 在屏幕上绘制预测注视点 (白点)
                    cv2.circle(screen_display, (draw_x, draw_y), 20, (255, 255, 255), -1)

                    # (可选) 在左上角显示当前坐标
                    cv2.putText(screen_display, f"Gaze: ({draw_x}, {draw_y})", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(screen_display, "No Face Detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(window_name, screen_display)

                if cv2.waitKey(1) & 0xFF == 27:  # 27 是 ESC 键的 ASCII 码
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()