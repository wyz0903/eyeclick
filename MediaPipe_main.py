"""
面部特征点检测测试与调试脚本。

该脚本调用 face_landmark_detector.py 中的组件，
打开电脑前置摄像头，实时进行面部特征点检测，并将结果渲染到屏幕上。
同时在左上角显示实时的 FPS (每秒帧数) 供性能调试。

运行方法:
    python test_camera.py
"""

import cv2
import time
from face_feature_MediaPipe import FaceLandmarkDetector


def main():
    """主程序：初始化摄像头和检测器，并运行实时循环。"""

    # 初始化视频捕获对象，0 通常代表默认的前置摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查设备连接或权限。")
        return

    print("提示：按 'q' 键或 'ESC' 键退出程序。")

    # 记录上一帧的时间，用于计算 FPS
    prev_time = 0

    # 使用上下文管理器 (with 语句) 初始化检测器，确保程序结束时资源自动释放
    with FaceLandmarkDetector(
            max_num_faces=1,
            refine_landmarks=True
    ) as detector:

        while True:
            # 读取一帧图像
            success, frame = cap.read()
            if not success:
                print("警告：忽略空的摄像机帧。")
                continue

            # 水平翻转图像，使显示画面表现为镜像（更符合用户的直觉）
            frame = cv2.flip(frame, 1)

            # 调用组件：处理图像并获取结果
            results = detector.process_frame(frame)

            # 测试：提取像素坐标（下游任务处理演示，当前脚本不实际使用它，只做打印或断点观察）
            # landmarks_list = detector.extract_landmarks(results, frame.shape)
            # if landmarks_list:
            #     print(f"检测到 {len(landmarks_list)} 张人脸，第一张人脸特征点数量：{len(landmarks_list[0])}")

            # 调用组件：将特征点渲染回图像帧上
            annotated_frame = detector.draw_landmarks(frame, results)

            # 计算并绘制 FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            cv2.putText(
                annotated_frame,
                f'FPS: {int(fps)}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # 显示结果
            cv2.imshow('Face Landmark Detection Test', annotated_frame)

            # 监听按键输入，按 'q' (键码113) 或 'ESC' (键码27) 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("退出程序...")
                break

    # 释放摄像头资源并关闭所有 OpenCV 窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()