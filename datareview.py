"""
HDF5 数据集可视化查看工具 (H5 Dataset Viewer)

该脚本用于读取并可视化 gaze_dataset.h5 中的数据。
可以直观地看到每次采样时保存的双眼图像，以及对应的目标坐标和头部特征。

操作说明:
    - 按 'D' 键或 '->' (右方向键) : 下一帧
    - 按 'A' 键或 '<-' (左方向键) : 上一帧
    - 按 'ESC' 或 'Q' 键 : 退出
"""

import h5py
import cv2
import numpy as np
import os

# ================= 配置区域 =================
DATASET_FILE = "gaze_dataset.h5"
DISPLAY_SCALE = 4  # 将 64x36 的眼睛图像放大显示的倍数，方便观察


# ============================================

def main():
    if not os.path.exists(DATASET_FILE):
        print(f"错误: 找不到数据集文件 '{DATASET_FILE}'。请先运行数据收集脚本。")
        return

    print(f"正在打开数据集: {DATASET_FILE} ...")

    # 以只读模式打开 HDF5 文件
    with h5py.File(DATASET_FILE, 'r') as f:
        # 获取数据集引用
        eye_l_ds = f['eye_l']
        eye_r_ds = f['eye_r']
        features_ds = f['features']
        targets_ds = f['targets']

        total_samples = targets_ds.shape[0]
        print(f"数据集加载成功！共包含 {total_samples} 条数据记录。")
        print("操作说明: 按 'D' 看下一帧，'A' 看上一帧，'Q' 退出。")

        if total_samples == 0:
            print("数据集为空！")
            return

        current_idx = 0
        window_name = "HDF5 Dataset Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # 读取当前索引的数据
            # HDF5 支持切片操作，类似于 numpy 数组
            img_l = eye_l_ds[current_idx]
            img_r = eye_r_ds[current_idx]
            features = features_ds[current_idx]
            target_x, target_y = targets_ds[current_idx]

            # 解析部分核心特征 (前 10 维是我们在收集脚本里组合的高级特征)
            head_pitch, head_yaw, head_roll = features[0:3]
            head_tx, head_ty, head_tz = features[3:6]
            l_iris_rel_x, l_iris_rel_y = features[6:8]
            r_iris_rel_x, r_iris_rel_y = features[8:10]

            # --- 图像可视化处理 ---
            # 放大双眼图像
            h, w = img_l.shape[:2]
            scaled_w, scaled_h = w * DISPLAY_SCALE, h * DISPLAY_SCALE
            img_l_disp = cv2.resize(img_l, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            img_r_disp = cv2.resize(img_r, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

            # 将左右眼水平拼接在一起，中间加一条黑色分割线
            separator = np.zeros((scaled_h, 10, 3), dtype=np.uint8)
            eyes_concat = np.hstack((img_l_disp, separator, img_r_disp))

            # 创建一个信息面板，用于显示数值
            panel_height = 200
            info_panel = np.zeros((panel_height, eyes_concat.shape[1], 3), dtype=np.uint8)

            # 在信息面板上绘制文字
            texts = [
                f"Sample Index: {current_idx + 1} / {total_samples}",
                f"Target (Screen): X={target_x:.0f}, Y={target_y:.0f}",
                f"Head Pose: Pitch={head_pitch:+.1f}, Yaw={head_yaw:+.1f}, Roll={head_roll:+.1f}",
                f"L Eye Rel: X={l_iris_rel_x:+.2f}, Y={l_iris_rel_y:+.2f}",
                f"R Eye Rel: X={r_iris_rel_x:+.2f}, Y={r_iris_rel_y:+.2f}"
            ]

            y_offset = 30
            for text in texts:
                cv2.putText(info_panel, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
                y_offset += 35

            # 将眼睛图像和信息面板垂直拼接
            final_display = np.vstack((eyes_concat, info_panel))

            # 显示画面
            cv2.imshow(window_name, final_display)

            # 键盘控制逻辑
            key = cv2.waitKey(0) & 0xFF

            if key == 27 or key == ord('q'):  # ESC 或 Q
                break
            elif key == ord('d') or key == 83:  # D 或 右方向键 (Windows OpenCV keyCode=83)
                current_idx = min(current_idx + 1, total_samples - 1)
            elif key == ord('a') or key == 81:  # A 或 左方向键 (Windows OpenCV keyCode=81)
                current_idx = max(current_idx - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()