"""
============================================================
Step 1 / 5 - Pose landmark detection (single patient)
============================================================
Backend usage:
    from step1_landmark_detection import LandmarkDetector

    detector = LandmarkDetector(MODEL_PATH)         # load once at startup
    keypoints_df = detector.detect(video_path)      # per request

Optional outputs (set paths to None to skip):
    keypoints_df = detector.detect(
        video_path,
        save_csv_path=...,            # save keypoints CSV
        save_annotated_video=...,     # save skeleton-overlay video
    )

Returns
-------
keypoints_df : pd.DataFrame with columns
    frame, landmark_id, x, y, z, visibility
============================================================
"""
import os
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Pose connections used for the skeleton overlay
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (9, 10),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
]


def _draw_landmarks(rgb_image, result):
    img = np.copy(rgb_image)
    h, w, _ = img.shape
    for pose_landmarks in result.pose_landmarks:
        for s, e in POSE_CONNECTIONS:
            if s < len(pose_landmarks) and e < len(pose_landmarks):
                x1 = int(pose_landmarks[s].x * w); y1 = int(pose_landmarks[s].y * h)
                x2 = int(pose_landmarks[e].x * w); y2 = int(pose_landmarks[e].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        for lm in pose_landmarks:
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 6, (255, 255, 0), -1)
    return img


class LandmarkDetector:
    """
    Wraps the MediaPipe pose-landmark model.
    Hold one instance for the lifetime of the backend process.
    """

    def __init__(self, model_path, num_poses=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
        self.model_path = model_path
        self.num_poses = num_poses
        self._base_options = python.BaseOptions(model_asset_path=model_path)

    def _new_landmarker(self):
        """A fresh landmarker is required per video to reset VIDEO-mode state."""
        options = vision.PoseLandmarkerOptions(
            base_options=self._base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
        )
        return vision.PoseLandmarker.create_from_options(options)

    # --------------------------------------------------------
    # Main entry: process one video, return keypoints DataFrame
    # --------------------------------------------------------
    def detect(self, video_path,
               save_csv_path=None,
               save_annotated_video=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Optional: annotated video writer
        writer = None
        if save_annotated_video:
            os.makedirs(os.path.dirname(save_annotated_video) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(save_annotated_video, fourcc, fps, (w, h))

        rows = []
        with self._new_landmarker() as landmarker:
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(1000 * frame_id / fps) if fps > 0 else frame_id
                result = landmarker.detect_for_video(mp_img, timestamp)

                # Collect keypoints (first detected pose only)
                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]
                    for i, lm in enumerate(pose):
                        rows.append([frame_id, i, lm.x, lm.y, lm.z, lm.visibility])

                # Optional annotated frame
                if writer is not None:
                    if result.pose_landmarks:
                        annotated = _draw_landmarks(rgb, result)
                        out_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    else:
                        out_frame = frame
                    writer.write(out_frame)

                frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()

        df = pd.DataFrame(rows, columns=[
            "frame", "landmark_id", "x", "y", "z", "visibility"
        ])

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
            df.to_csv(save_csv_path, index=False)

        return df


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# Edit the paths below and run:  python step1_landmark_detection.py
# Delete this whole section for production - nothing above depends on it.
# ============================================================
if __name__ == "__main__":
    DEBUG_MODEL_PATH = "a_model-for-landmark\pose_landmarker_heavy.task"
    DEBUG_VIDEO_PATH = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\Patient1_leg-agility-L_1.mp4"
    DEBUG_OUTPUT_DIR = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\leg-agility\a_landmark-detection"

    video_name = os.path.splitext(os.path.basename(DEBUG_VIDEO_PATH))[0]
    out_dir    = os.path.join(DEBUG_OUTPUT_DIR, video_name)
    csv_out    = os.path.join(out_dir, f"{video_name}_keypoints.csv")
    video_out  = os.path.join(out_dir, f"{video_name}_annotated.mp4")

    print(f"[DEBUG] Loading model: {DEBUG_MODEL_PATH}")
    detector = LandmarkDetector(DEBUG_MODEL_PATH)

    print(f"[DEBUG] Processing : {DEBUG_VIDEO_PATH}")
    df = detector.detect(
        DEBUG_VIDEO_PATH,
        save_csv_path=csv_out,
        save_annotated_video=video_out,
    )
    print(f"[DEBUG] Got {len(df)} keypoint rows ({df['frame'].nunique()} frames)")
    print(f"[DEBUG] Keypoints CSV: {csv_out}")
    print(f"[DEBUG] Annotated MP4: {video_out}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================