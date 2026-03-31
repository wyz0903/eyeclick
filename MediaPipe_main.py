"""
面部特征与视线可视化测试脚本。
"""

import cv2
import numpy as np
from face_feature_MediaPipe import FaceLandmarkDetector

def draw_head_pose_axes(frame, rvec, tvec, camera_matrix, dist_coeffs):
    """在鼻尖绘制表示头部朝向的 3D 坐标轴"""
    axis = np.float32([[500, 0, 0], [0, 500, 0], [0, 0, 500]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    origin, _ = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(np.int32(origin).reshape(2))

    frame = cv2.line(frame, origin, tuple(imgpts[0]), (0, 0, 255), 3) # X轴: Pitch
    frame = cv2.line(frame, origin, tuple(imgpts[1]), (0, 255, 0), 3) # Y轴: Yaw
    frame = cv2.line(frame, origin, tuple(imgpts[2]), (255, 0, 0), 3) # Z轴: Roll
    return frame

def draw_gaze_lines(frame, features, landmarks, h, w):
    """根据眼部相对特征绘制高灵敏度的视线射线"""
    # 屏幕右侧眼睛 (用户左眼) 的近似中心
    l_center = np.array([(landmarks[133].x + landmarks[33].x) / 2 * w,
                         (landmarks[159].y + landmarks[145].y) / 2 * h])

    # 屏幕左侧眼睛 (用户右眼) 的近似中心
    r_center = np.array([(landmarks[263].x + landmarks[362].x) / 2 * w,
                         (landmarks[386].y + landmarks[374].y) / 2 * h])

    # 放大系数，控制箭头的长度和灵敏度
    # 因为现在的 rel_x 和 rel_y 在中心时严格等于 0，我们可以放心大胆地放大
    multiplier = 150

    # 计算射线终点：由于 rel_x 和 rel_y 正值分别代表向右和向下，直接加到中心坐标上即可
    l_gaze_end = (int(l_center[0] + features['l_iris_rel_x'] * multiplier),
                  int(l_center[1] + features['l_iris_rel_y'] * multiplier))

    r_gaze_end = (int(r_center[0] + features['r_iris_rel_x'] * multiplier),
                  int(r_center[1] + features['r_iris_rel_y'] * multiplier))

    l_center_tuple = (int(l_center[0]), int(l_center[1]))
    r_center_tuple = (int(r_center[0]), int(r_center[1]))

    # 绘制黄色的视线箭头
    cv2.arrowedLine(frame, l_center_tuple, l_gaze_end, (0, 255, 255), 2, tipLength=0.3)
    cv2.arrowedLine(frame, r_center_tuple, r_gaze_end, (0, 255, 255), 2, tipLength=0.3)

    # 绘制瞳孔中心点（红点），方便核对
    l_iris_actual = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    r_iris_actual = (int(landmarks[473].x * w), int(landmarks[473].y * h))
    cv2.circle(frame, l_iris_actual, 2, (0, 0, 255), -1)
    cv2.circle(frame, r_iris_actual, 2, (0, 0, 255), -1)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    with FaceLandmarkDetector(max_num_faces=1, refine_landmarks=True) as detector:
        while True:
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = detector.process_frame(frame)
            # 可选：注释掉下一行可以关闭绿色网格显示，让画面更干净
            # annotated_frame = detector.draw_landmarks(frame.copy(), results)
            annotated_frame = frame.copy()

            features = detector.extract_comprehensive_features(results, (h, w))

            if features and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 绘制头部坐标轴
                camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
                dist_coeffs = np.zeros((4, 1))
                annotated_frame = draw_head_pose_axes(
                    annotated_frame, features["rvec"], features["tvec"], camera_matrix, dist_coeffs
                )

                # 绘制视线
                annotated_frame = draw_gaze_lines(annotated_frame, features, landmarks, h, w)

                # 数值面板
                y_pos = 30
                texts = [
                    f"Head Pitch: {features['head_pitch']:+05.1f}",
                    f"Head Yaw  : {features['head_yaw']:+05.1f}",
                    f"Head Roll : {features['head_roll']:+05.1f}",
                    f"Eye L Rel : X {features['l_iris_rel_x']:+0.2f}  Y {features['l_iris_rel_y']:+0.2f}",
                    f"Eye R Rel : X {features['r_iris_rel_x']:+0.2f}  Y {features['r_iris_rel_y']:+0.2f}",
                ]

                cv2.rectangle(annotated_frame, (10, 10), (380, 180), (0, 0, 0), -1)
                for txt in texts:
                    # 当眼球坐标偏离中心超过 0.1 时，将字体颜色变绿以示反馈
                    color = (255, 255, 255)
                    if "Eye" in txt:
                        rel_x = float(txt.split("X ")[1].split(" ")[0])
                        rel_y = float(txt.split("Y ")[1])
                        if abs(rel_x) > 0.15 or abs(rel_y) > 0.15:
                            color = (0, 255, 0)

                    cv2.putText(annotated_frame, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 30

            cv2.imshow('Robust Features & Gaze Visualizer', annotated_frame)

            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()