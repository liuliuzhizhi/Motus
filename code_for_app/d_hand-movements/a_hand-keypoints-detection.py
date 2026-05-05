"""
============================================================
手指轻叩 Step 1 / 5 - 手部关键点检测 (单视频)
============================================================
后端使用方式:
    from step1_hand_landmark_detection import HandLandmarkDetector

    detector = HandLandmarkDetector(MODEL_PATH)         # 启动时加载一次
    keypoints_df = detector.detect(video_path)          # 每个请求

返回 DataFrame 列:
    frame, hand_label, landmark_id, x, y, z, x_world, y_world, z_world
============================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 绘图样式 (仅用于 save_annotated_video)
# ============================================================
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

mp_hands_module = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils


def _draw_landmarks_on_frame(rgb_image, detection_result):
    """在 RGB 帧上画手部骨架, 返回标注后的 RGB 图。"""
    landmark_style = mp_drawing.DrawingSpec(color=(70, 206, 255),
                                            thickness=3, circle_radius=0)
    connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    annotated = np.copy(rgb_image)

    for hand_landmarks, handedness in zip(detection_result.hand_landmarks,
                                          detection_result.handedness):
        mp_drawing.draw_landmarks(
            annotated, hand_landmarks,
            mp_hands_module.HAND_CONNECTIONS,
            landmark_style, connection_style,
        )
        h, w, _ = annotated.shape
        text_x = int(min(lm.x for lm in hand_landmarks) * w)
        text_y = int(min(lm.y for lm in hand_landmarks) * h) - MARGIN
        cv2.putText(annotated, handedness[0].category_name,
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR,
                    FONT_THICKNESS, cv2.LINE_AA)
    return annotated


# ============================================================
# 检测器类
# ============================================================
class HandLandmarkDetector:
    """
    封装 MediaPipe 手部关键点模型。
    后端启动时实例化一次, 之后每个请求调 detect()。
    HandLandmarker 是 IMAGE 模式, 多个视频可以复用同一个实例,
    不需要像 PoseLandmarker 那样每个视频重建。
    """

    def __init__(self, model_path, num_hands=2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到手部模型文件: {model_path}")
        self.model_path = model_path
        self.num_hands = num_hands
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=num_hands,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, video_path,
               save_csv_path=None,
               save_annotated_video=None):
        """
        在视频上跑手部关键点检测。

        Parameters
        ----------
        video_path : str
        save_csv_path : str or None
        save_annotated_video : str or None

        Returns
        -------
        pd.DataFrame, 列见模块文档
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_annotated_video:
            os.makedirs(os.path.dirname(save_annotated_video) or ".",
                        exist_ok=True)
            writer = cv2.VideoWriter(
                save_annotated_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h),
            )

        records = []
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._detector.detect(mp_image)

            # 同时记录归一化坐标 (lm) 和世界坐标 (w_lm)
            if result.hand_landmarks and result.hand_world_landmarks:
                for landmarks, world_landmarks, handedness in zip(
                        result.hand_landmarks,
                        result.hand_world_landmarks,
                        result.handedness):
                    hand_label = handedness[0].category_name
                    for lm_id, (lm, w_lm) in enumerate(zip(landmarks, world_landmarks)):
                        records.append([
                            frame_idx, hand_label, lm_id,
                            lm.x, lm.y, lm.z,
                            w_lm.x, w_lm.y, w_lm.z,
                        ])

            if writer is not None:
                if result.hand_landmarks:
                    annotated = _draw_landmarks_on_frame(frame_rgb, result)
                    out_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                else:
                    out_frame = frame_bgr
                writer.write(out_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        df = pd.DataFrame(records, columns=[
            "frame", "hand_label", "landmark_id",
            "x", "y", "z", "x_world", "y_world", "z_world",
        ])

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
            df.to_csv(save_csv_path, index=False)

        return df


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# ============================================================
if __name__ == "__main__":
    DEBUG_MODEL_PATH = "a_model-for-landmark\hand_landmarker.task"
    DEBUG_VIDEO_PATH = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\Patient48_hand-movements-L_3.mp4"
    DEBUG_OUTPUT_DIR = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\a_landmark-detection"

    video_name = os.path.splitext(os.path.basename(DEBUG_VIDEO_PATH))[0]
    out_dir   = os.path.join(DEBUG_OUTPUT_DIR, video_name)
    csv_out   = os.path.join(out_dir, f"{video_name}_hand_landmarks.csv")
    video_out = os.path.join(out_dir, f"{video_name}_annotated.mp4")

    print(f"[DEBUG] 加载模型: {DEBUG_MODEL_PATH}")
    detector = HandLandmarkDetector(DEBUG_MODEL_PATH, num_hands=2)

    print(f"[DEBUG] 处理视频: {DEBUG_VIDEO_PATH}")
    df = detector.detect(
        DEBUG_VIDEO_PATH,
        save_csv_path=csv_out,
        save_annotated_video=video_out,
    )
    print(f"[DEBUG] 收到 {len(df)} 行 ({df['frame'].nunique()} 帧)")
    print(f"[DEBUG] 关键点 CSV: {csv_out}")
    print(f"[DEBUG] 标注视频  : {video_out}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================