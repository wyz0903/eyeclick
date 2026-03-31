import cv2
import time
from face_feature_Face_Alignment import FaceLandmarkDetector

def main():
    """
    主测试函数。
    负责打开前置摄像头，读取视频流，调用组件进行处理，并实时显示帧率(FPS)和结果。
    """
    # 1. 实例化组件
    # 注意：如果您的电脑配有 NVIDIA 显卡且已配置好 CUDA 环境，请将 device='cpu' 更改为 device='cuda'
    # 这样可以大幅提升实时处理的帧率。
    detector = FaceLandmarkDetector(enable_3d=True, device='cuda', face_detector='sfd')

    # 2. 打开前置摄像头 (0 通常是系统默认的前置摄像头)
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (ID: {camera_id})。请检查设备连接。")
        return

    print("[INFO] 摄像头已打开。按 'q' 键退出测试。")

    # 用于计算 FPS 的变量
    prev_time = time.time()

    try:
        while True:
            # 3. 读取一帧图像
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 无法从摄像头读取画面，跳过此帧。")
                continue

            # 4. 镜像翻转图像 (前置摄像头通常需要镜像处理，符合用户直觉)
            frame = cv2.flip(frame, 1)

            # 5. 调用组件核心功能：检测并绘制特征点
            # 这里调用了高级封装 API process_and_draw
            processed_frame, landmarks = detector.process_and_draw(frame)

            # 6. 计算并绘制 FPS (性能监控)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 在图像左上角显示 FPS 信息
            cv2.putText(
                processed_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255), # 红色字体
                2
            )

            # 并在屏幕上提示检测到的人脸数量
            face_count = len(landmarks) if landmarks else 0
            cv2.putText(
                processed_frame,
                f"Faces: {face_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0), # 蓝色字体
                2
            )

            # 7. 实时显示结果
            cv2.imshow("Face Landmark Detector Test", processed_frame)

            # 8. 监听按键事件，按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 收到退出指令，正在关闭...")
                break

    except KeyboardInterrupt:
        print("[INFO] 强制中断，正在清理资源...")
    except Exception as e:
        print(f"[ERROR] 运行时发生异常: {e}")
    finally:
        # 9. 资源清理阶段 (防御性编程，确保资源得到释放)
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 资源已释放。测试结束。")

if __name__ == "__main__":
    main()