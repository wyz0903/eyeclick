"""
面部特征点检测组件测试脚本 (Test Script for Face Landmark Detector)

该脚本负责打开电脑前置摄像头，读取实时视频流，
并调用 face_landmark_detector.py 中的类来进行实时的人脸网格检测和显示。

使用方法:
    直接运行此脚本即可开启测试界面： python test_camera.py
    按下 'q' 键或 'ESC' 键退出程序。
"""

import cv2
import sys
import time
# 从我们封装好的组件模块中导入检测器类
from face_feature import FaceLandmarkDetector


def main():
    print("正在初始化摄像头...")
    # 尝试打开默认摄像头 (索引为 0 的前置摄像头)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误: 无法打开前置摄像头。请检查摄像头是否被占用或连接是否正常。")
        sys.exit(1)

    # 尝试设置摄像头分辨率 (可选，可以根据需求注释掉)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("摄像头初始化成功。正在加载面部特征点检测模型...")
    # 实例化我们的组件类
    # 使用针对视频流优化的默认参数 (static_image_mode=False)
    detector = FaceLandmarkDetector(
        max_num_faces=1,  # 这里我们设置为只检测一张脸作为测试
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("模型加载完成。开始实时检测。按下 'q' 或 'ESC' 键退出...")

    # 用于计算 FPS (帧率)
    prev_time = 0

    try:
        while True:
            # 1. 读取摄像头帧
            success, frame = cap.read()
            if not success:
                print("警告: 忽略空白摄像头帧。")
                continue

            # 建议将前置摄像头的画面水平翻转，以便实现“镜像”效果，更符合直觉
            frame = cv2.flip(frame, 1)

            # 2. 将帧传递给组件进行检测
            results = detector.detect_landmarks(frame)

            # 3. 在画面上绘制特征点 (可视化)
            annotated_frame = detector.draw_landmarks(frame, results)

            # (可选测试) 演示如何获取具体的坐标点列表
            # coords = detector.extract_landmark_coordinates(frame, results)
            # if coords:
            #     print(f"检测到人脸，特征点数量: {len(coords[0])}")

            # 4. 计算并显示 FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(
                annotated_frame, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 5. 显示最终处理后的画面
            cv2.imshow('Face Landmark Detection Test', annotated_frame)

            # 6. 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            # 按 'q' 或 ESC 键退出
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n检测到键盘中断，准备退出...")
    finally:
        # 7. 优雅地清理和释放资源
        print("正在释放资源...")
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == "__main__":
    main()