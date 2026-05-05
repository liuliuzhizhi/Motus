"""
============================================================
手指轻叩 Step 2 / 5 - 左右手筛选 + 距离信号生成 (单视频)
============================================================
后端使用方式:
    from step2_hand_signal import HandSignalExtractor

    extractor = HandSignalExtractor(face_model_path)             # 启动时加载一次
    
    # 获取右手信号
    signal_df_R = extractor.extract(keypoints_df, video_path, target_hand="right")
    
    # 获取左手信号
    signal_df_L = extractor.extract(keypoints_df, video_path, target_hand="left")

返回 DataFrame 列:
    frame, is_valid, distance_4_8

注意: 这一步需要"原始视频路径", 因为只检测到一只手的某些帧需要用人脸
位置辅助判定是哪只手。纯关键点不够。
============================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# 工具函数 (严格保持原样)
# ============================================================
def _euclidean_distance(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def _compute_valid_frames(frame_list, total_frames):
    """
    最终版规则:
    1. 连续 >= 3 帧存在 → 启动有效 (这 3 帧都有效)
    2. 连续 >= 3 帧缺失 → 终止有效 (这 3 帧都无效)
    3. 没有连续 3 帧缺失就维持有效

    返回 dict[frame_id -> 0/1]
    """
    if total_frames == 0:
        return {}
    frame_set = set(frame_list)
    full_frames = list(range(total_frames))
    valid_dict = {f: 0 for f in full_frames}

    consecutive_exist = 0
    consecutive_missing = 0
    is_valid = False

    for i, f in enumerate(full_frames):
        if f in frame_set:
            consecutive_exist += 1
            consecutive_missing = 0
        else:
            consecutive_missing += 1
            consecutive_exist = 0

        # 启动有效
        if not is_valid and consecutive_exist >= 3:
            is_valid = True
            for j in range(3):
                valid_dict[full_frames[i - j]] = 1

        # 终止有效
        if is_valid and consecutive_missing >= 3:
            is_valid = False
            for j in range(3):
                valid_dict[full_frames[i - j]] = 0

        if is_valid:
            valid_dict[f] = 1

    return valid_dict


# ============================================================
# 主类
# ============================================================
class HandSignalExtractor:
    """
    把手部关键点 + 视频, 转成 (frame, is_valid, distance_4_8) 的距离信号。

    "右手"的判定原则 (以 target_hand="right" 为例):
      - 同帧检测到两只手: 取 x 均值较小的那只 (画面中靠左, 即被试视角的右手)
      - 只检测到一只手且 label=Left: 直接当成右手 (label 反了)
      - 只检测到一只手且 label=Right: 用人脸检测判断, 在脸左侧(画面靠左)才认
        (人脸检测每 5 帧做一次以节省时间)
        
    "左手"的判定原则 (以 target_hand="left" 为例):
      - 逻辑完全镜像相反。

    `LEFT_OR_RIGHT` 配置: 决定双手时取 min(靠左屏幕) 还是 max(靠右屏幕)。
    默认 right_in_view (取屏幕较左者, 适合面对镜头的患者右手)。
    """

    def __init__(self, face_model_path,
                 face_check_interval=5,
                 hand_select="right_in_view"):
        """
        Parameters
        ----------
        face_model_path : str
            BlazeFace 模型路径 (.tflite)
        face_check_interval : int
            每多少帧重新跑一次人脸检测 (其余帧用缓存)
        hand_select : {"right_in_view", "left_in_view"}
            "right_in_view"  : 找患者右手时取 x 均值最小者 (画面较左)
            "left_in_view"   : 找患者右手时取 x 均值最大者 (画面较右)
        """
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(f"找不到人脸模型: {face_model_path}")
        if hand_select not in ("right_in_view", "left_in_view"):
            raise ValueError("hand_select must be 'right_in_view' or 'left_in_view'")

        self.face_model_path = face_model_path
        self.face_check_interval = int(face_check_interval)
        self.hand_select = hand_select

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.IMAGE,
        )
        self._face_detector = FaceDetector.create_from_options(options)

    # --------------------------------------------------------
    def extract(self, keypoints_df, video_path,
                target_hand="right",  # 支持 "left"
                save_csv_path=None):
        """
        Parameters
        ----------
        keypoints_df : pd.DataFrame
            Step 1 产物。必需列: frame, hand_label, landmark_id, x, y
        video_path : str
            原始视频, 用于人脸辅助判定
        target_hand : {"left", "right"}
            指定要提取左手还是右手的特征
        save_csv_path : str or None

        Returns
        -------
        pd.DataFrame  列: frame, is_valid, distance_4_8
        """
        target_hand = target_hand.lower()
        if target_hand not in ("left", "right"):
            raise ValueError("target_hand must be 'left' or 'right'")

        if keypoints_df is None or len(keypoints_df) == 0:
            return pd.DataFrame(columns=["frame", "is_valid", "distance_4_8"])
        for c in ("frame", "hand_label", "landmark_id", "x", "y"):
            if c not in keypoints_df.columns:
                raise ValueError(f"keypoints_df 缺列: {c}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        df = keypoints_df[["frame", "hand_label", "landmark_id", "x", "y"]]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ---- 第一阶段: 选出每帧的目标手 ----
        target_hand_data = {}            # frame -> 该帧的目标手关键点 DataFrame
        target_hand_frames = []
        face_center_x_cache = None
        single_hand_counter = 0          # 原为 left_only_counter

        for frame, group in df.groupby("frame"):
            hands = group.groupby("hand_label")
            chosen = None

            if len(hands) == 2:
                # 两只手: 取 x 均值较小/较大者
                hand_means = {name: g["x"].mean() for name, g in hands}
                
                # ---- 【修改逻辑】适配左右手双手的选择 ----
                if target_hand == "right":
                    if self.hand_select == "right_in_view":
                        chosen = min(hand_means, key=hand_means.get)
                    else:
                        chosen = max(hand_means, key=hand_means.get)
                else: # target_hand == "left"
                    # 左手的选择逻辑镜像相反
                    if self.hand_select == "right_in_view":
                        chosen = max(hand_means, key=hand_means.get)
                    else:
                        chosen = min(hand_means, key=hand_means.get)
                        
            else:
                only_hand = list(hands.groups.keys())[0]
                
                # ---- 【修改逻辑】适配左右手单手人脸校验逻辑 ----
                if target_hand == "right":
                    if only_hand == "Left":
                        # 单 Left -> 直接当成右手 (label 反了)
                        chosen = only_hand
                    else: #如果MediaPipe说是右手
                        # 单 Right -> 需要人脸检测确认
                        single_hand_counter += 1
                        hand_center_x = float(group["x"].mean())

                        need_face = (
                            face_center_x_cache is None
                            or single_hand_counter % self.face_check_interval == 1
                        )
                        if need_face:
                            face_center_x_cache = self._detect_face_center_x(cap, frame)

                        if face_center_x_cache == -1:
                            # 没检测到人脸 -> 信任 only_hand
                            chosen = only_hand
                        elif face_center_x_cache <= hand_center_x:
                            # 手在脸左侧 (画面靠左 = 患者右手) -> 认
                            chosen = only_hand
                        else:
                            continue
                            
                else: # target_hand == "left"
                    if only_hand == "Right":
                        # 单 Right -> 直接当成左手 (label 反了)
                        chosen = only_hand
                    else:
                        # 单 Left -> 需要人脸检测确认
                        single_hand_counter += 1
                        hand_center_x = float(group["x"].mean())

                        need_face = (
                            face_center_x_cache is None
                            or single_hand_counter % self.face_check_interval == 1
                        )
                        if need_face:
                            face_center_x_cache = self._detect_face_center_x(cap, frame)

                        if face_center_x_cache == -1:
                            # 没检测到人脸 -> 信任 only_hand
                            chosen = only_hand
                        elif face_center_x_cache >= hand_center_x:
                            # 手在脸右侧 (画面靠右 = 患者左手) -> 认
                            chosen = only_hand
                        else:
                            continue

            if chosen is not None:
                target_hand_data[frame] = hands.get_group(chosen)
                target_hand_frames.append(frame)

        cap.release()

        # ---- 第二阶段: 计算每帧 is_valid ----
        valid_dict = _compute_valid_frames(target_hand_frames, total_frames)
        if not valid_dict:
            return pd.DataFrame(columns=["frame", "is_valid", "distance_4_8"])

        # ---- 第三阶段: 输出连续帧, 计算 4-8 距离 ----
        rows = []
        min_frame = min(valid_dict.keys())
        max_frame = max(valid_dict.keys())
        for frame in range(min_frame, max_frame + 1):
            is_valid = valid_dict.get(frame, 0)
            distance = np.nan
            if frame in target_hand_data and is_valid == 1:
                g = target_hand_data[frame]
                kp4 = g[g["landmark_id"] == 4]
                kp8 = g[g["landmark_id"] == 8]
                if len(kp4) > 0 and len(kp8) > 0:
                    x1, y1 = kp4.iloc[0][["x", "y"]]
                    x2, y2 = kp8.iloc[0][["x", "y"]]
                    distance = _euclidean_distance(x1, y1, x2, y2)
            rows.append([frame, is_valid, distance])

        out = pd.DataFrame(rows, columns=["frame", "is_valid", "distance_4_8"])

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
            out.to_csv(save_csv_path, index=False)

        return out

    # --------------------------------------------------------
    def _detect_face_center_x(self, cap, frame_idx):
        """从视频里读出 frame_idx 帧, 跑一次人脸检测, 返回脸中心 x 像素坐标。"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_image = cap.read()
        if not ret:
            return -1
        rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_detector.detect(mp_image)
        if not result.detections:
            return -1
        bbox = result.detections[0].bounding_box
        return bbox.origin_x + bbox.width / 2

# 为兼容旧代码调用，保留原类名别名
RightHandSignalExtractor = HandSignalExtractor


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# ============================================================
if __name__ == "__main__":
    DEBUG_FACE_MODEL = r"a_model-for-landmark\blaze_face_short_range.tflite"
    DEBUG_KEYPOINTS_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\a_finger-tapping\a_landmark-detection\Patient13_finger-tapping-L_1\Patient13_finger-tapping-L_1_hand_landmarks.csv"
    DEBUG_VIDEO_PATH = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\a_finger-tapping\a_landmark-detection\Patient13_finger-tapping-L_1\Patient13_finger-tapping-L_1_annotated.mp4"
    DEBUG_OUTPUT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\a_finger-tapping\b_kinematic-signal\debug_distance_4_8.csv"

    print(f"[DEBUG] 读取关键点: {DEBUG_KEYPOINTS_CSV}")
    keypoints = pd.read_csv(DEBUG_KEYPOINTS_CSV)

    print(f"[DEBUG] 加载人脸模型: {DEBUG_FACE_MODEL}")
    extractor = HandSignalExtractor(DEBUG_FACE_MODEL)

    print(f"[DEBUG] 提取距离信号 (视频={DEBUG_VIDEO_PATH})")
    # 只需要在 extract 中传入 target_hand="right" 即可
    signal = extractor.extract(
        keypoints, DEBUG_VIDEO_PATH,
        target_hand="left",   #### 注意参数，选择左右手
        save_csv_path=DEBUG_OUTPUT_CSV,
    )
    print(f"[DEBUG] 共 {len(signal)} 帧, 有效 {int(signal['is_valid'].sum())} 帧")
    print(f"[DEBUG] 信号 CSV: {DEBUG_OUTPUT_CSV}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================