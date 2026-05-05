"""
============================================================
开合手 (Hand-Movements) Step 2 / 5 - 目标手筛选 + 多边形面积信号
============================================================
后端使用方式:
    from handmove_step2_polygon_signal import HandMovementSignalExtractor

    # 启动时加载一次
    extractor = HandMovementSignalExtractor(face_model_path)

    # 每个请求
    signal_df = extractor.extract(
        keypoints_df, video_path,
        target_hand="left",          # "left" 或 "right"
    )

返回 DataFrame 列:
    frame, is_valid, polygon_area_0_1_4_8_12_16_20

注意: 这一步需要原始视频路径, 因为只检测到一只手时需要用人脸位置
辅助判定。纯 keypoints 不够。

与手指轻叩 step 2 的区别:
  - 手指: 计算两个关键点 (4 和 8) 之间的距离
  - 开合: 计算 7 个关键点 (0,1,4,8,12,16,20) 围成的多边形面积
============================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# 用于多边形面积的关键点 (掌心 + 5 个指尖 + 食指根)
# ============================================================
POLYGON_LANDMARK_IDS = [0, 1, 4, 8, 12, 16, 20]
SIGNAL_COLUMN_NAME   = "polygon_area_0_1_4_8_12_16_20"


# ============================================================
# 工具函数
# ============================================================
def _polygon_area(points):
    """鞋带公式 (shoelace) 计算多边形面积。少于 3 个点返回 NaN。"""
    if len(points) < 3:
        return np.nan
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _compute_valid_frames(frame_list, total_frames, con_num):
    """
    用 con_num 连续帧阈值生成 valid 标签:
      - 连续 >= con_num 帧存在 -> 启动有效 (回填这 con_num 帧)
      - 连续 >= con_num 帧缺失 -> 终止有效 (回填这 con_num 帧)
      - 期间维持有效

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

        if not is_valid and consecutive_exist >= con_num:
            is_valid = True
            for j in range(con_num):
                valid_dict[full_frames[i - j]] = 1

        if is_valid and consecutive_missing >= con_num:
            is_valid = False
            for j in range(con_num):
                valid_dict[full_frames[i - j]] = 0

        if is_valid:
            valid_dict[f] = 1

    return valid_dict


# ============================================================
# 主类
# ============================================================
class HandMovementSignalExtractor:
    """
    把手部关键点 + 视频, 转成 (frame, is_valid, polygon_area_*) 的面积信号。

    "目标手"判定原则 (target_hand 参数控制):
      target_hand="right" -> 患者右手 (画面较左)
        - 双手: 取 x 均值较小者
        - 单 Left: 直接采用 (label 反了)
        - 单 Right: 用人脸辅助, 手在脸右侧才认
      target_hand="left"  -> 患者左手 (画面较右)
        - 双手: 取 x 均值较大者
        - 单 Right: 直接采用
        - 单 Left:  用人脸辅助, 手在脸左侧才认

    人脸检测每 face_check_interval 帧跑一次 (用缓存), 减少开销。
    """

    def __init__(self, face_model_path,
                 face_check_interval: int = 5,
                 con_num: int = 6):
        """
        Parameters
        ----------
        face_model_path : str
            BlazeFace 模型路径 (.tflite)
        face_check_interval : int
            每多少帧重新跑一次人脸检测 (其余帧用缓存)
        con_num : int
            连续帧阈值 (启动/终止有效的帧数)
        """
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(f"找不到人脸模型: {face_model_path}")
        self.face_model_path = face_model_path
        self.face_check_interval = int(face_check_interval)
        self.con_num = int(con_num)

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
    def extract(self, keypoints_df: pd.DataFrame,
                video_path: str,
                target_hand: str = "right",
                save_csv_path=None) -> pd.DataFrame:
        """
        Parameters
        ----------
        keypoints_df : pd.DataFrame
            Step 1 产物。必需列: frame, hand_label, landmark_id, x, y
        video_path : str
            原始视频, 用于单手时的人脸辅助判定
        target_hand : {"left", "right"}
            想提取的患者手别
        save_csv_path : str or None

        Returns
        -------
        pd.DataFrame  列: frame, is_valid, polygon_area_0_1_4_8_12_16_20
        """
        target_hand = self._validate_target_hand(target_hand)

        if keypoints_df is None or len(keypoints_df) == 0:
            return pd.DataFrame(columns=["frame", "is_valid", SIGNAL_COLUMN_NAME])
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
        target_hand_data = {}        # frame -> 该帧目标手的关键点 DataFrame
        target_hand_frames = []
        face_center_x_cache = None
        single_hand_counter = 0

        for frame, group in df.groupby("frame"):
            hands = group.groupby("hand_label")
            chosen, face_center_x_cache, single_hand_counter = self._choose_target_hand(
                hands=hands,
                group=group,
                target_hand=target_hand,
                cap=cap,
                frame=frame,
                face_center_x_cache=face_center_x_cache,
                single_hand_counter=single_hand_counter,
            )
            if chosen is None:
                continue
            target_hand_data[frame] = hands.get_group(chosen)
            target_hand_frames.append(frame)

        cap.release()

        # ---- 第二阶段: 计算每帧 is_valid ----
        valid_dict = _compute_valid_frames(
            target_hand_frames, total_frames, self.con_num
        )
        if not valid_dict:
            return pd.DataFrame(columns=["frame", "is_valid", SIGNAL_COLUMN_NAME])

        # ---- 第三阶段: 输出连续帧, 计算多边形面积 ----
        rows = []
        min_frame = min(valid_dict.keys())
        max_frame = max(valid_dict.keys())
        for frame in range(min_frame, max_frame + 1):
            is_valid = valid_dict.get(frame, 0)
            area_value = np.nan
            if frame in target_hand_data and is_valid == 1:
                g = target_hand_data[frame]
                points = []
                for lid in POLYGON_LANDMARK_IDS:
                    kp = g[g["landmark_id"] == lid]
                    if len(kp) == 0:
                        points = []
                        break
                    x, y = kp.iloc[0][["x", "y"]]
                    points.append((x, y))
                if len(points) == len(POLYGON_LANDMARK_IDS):
                    area_value = _polygon_area(points)
            rows.append([frame, is_valid, area_value])

        out = pd.DataFrame(rows, columns=["frame", "is_valid", SIGNAL_COLUMN_NAME])

        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
            out.to_csv(save_csv_path, index=False)

        return out

    # --------------------------------------------------------
    # 内部: 单帧的目标手选择
    # --------------------------------------------------------
    def _choose_target_hand(self, hands, group, target_hand,
                            cap, frame, face_center_x_cache,
                            single_hand_counter):
        """返回 (chosen_hand_label, 更新后的人脸缓存, 更新后的单手计数)。"""
        # 双手: 直接按 x 均值取一侧
        if len(hands) == 2:
            hand_means = {name: g["x"].mean() for name, g in hands}
            if target_hand == "right":
                chosen = min(hand_means, key=hand_means.get)
            else:
                chosen = max(hand_means, key=hand_means.get)
            return chosen, face_center_x_cache, single_hand_counter

        # 单手: 看 label 决定是否需要人脸辅助
        only_hand = list(hands.groups.keys())[0]
        hand_center_x = float(group["x"].mean())

        # label 与目标手的"自然 label"相反 -> 直接采用 (label 经常反)
        # 自然 label: target_hand="right" 期望的 label 是 "Left" (镜像)
        natural_opposite = {"right": "Left", "left": "Right"}[target_hand]
        if only_hand == natural_opposite:
            return only_hand, face_center_x_cache, single_hand_counter

        # 否则需要人脸辅助
        single_hand_counter += 1
        need_face = (
            face_center_x_cache is None
            or single_hand_counter % self.face_check_interval == 1
        )
        if need_face:
            face_center_x_cache = self._detect_face_center_x(cap, frame)

        if face_center_x_cache == -1:
            # 没检测到人脸 -> 信任 only_hand
            return only_hand, face_center_x_cache, single_hand_counter

        # 几何判定 (与原代码完全一致):
        # target=right 时: hand_center_x >= face_center_x_cache 才接受
        # target=left  时: hand_center_x <= face_center_x_cache 才接受
        if target_hand == "right":
            if hand_center_x >= face_center_x_cache:
                return only_hand, face_center_x_cache, single_hand_counter
        else:
            if hand_center_x <= face_center_x_cache:
                return only_hand, face_center_x_cache, single_hand_counter

        return None, face_center_x_cache, single_hand_counter

    # --------------------------------------------------------
    def _detect_face_center_x(self, cap, frame_idx):
        """从视频读 frame_idx 帧, 跑一次人脸检测, 返回脸中心 x 像素坐标。"""
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

    @staticmethod
    def _validate_target_hand(target_hand):
        target_hand = str(target_hand).lower()
        if target_hand not in ("left", "right"):
            raise ValueError('target_hand 只能是 "left" 或 "right"')
        return target_hand


# ============================================================
# >>>>>>>>>>>>>>>>>>  DEBUG / DEV-ONLY  >>>>>>>>>>>>>>>>>>>>>>
# ============================================================
if __name__ == "__main__":
    DEBUG_FACE_MODEL = r"a_model-for-landmark\blaze_face_short_range.tflite"
    DEBUG_KEYPOINTS_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\a_landmark-detection\Patient48_hand-movements-L_3\Patient48_hand-movements-L_3_hand_landmarks.csv"
    DEBUG_VIDEO_PATH = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\a_landmark-detection\Patient48_hand-movements-L_3\Patient48_hand-movements-L_3_annotated.mp4"
    DEBUG_TARGET_HAND = "left"
    DEBUG_OUTPUT_CSV = r"F:\Chinese Excluded Files\Research\Clinical\develop\code\debug_for_app\c_hand-movements\b_kinematic-signal\debug_polygon_area.csv"

    print(f"[DEBUG] 读取关键点: {DEBUG_KEYPOINTS_CSV}")
    keypoints = pd.read_csv(DEBUG_KEYPOINTS_CSV)

    print(f"[DEBUG] 加载人脸模型: {DEBUG_FACE_MODEL}")
    extractor = HandMovementSignalExtractor(
        DEBUG_FACE_MODEL, face_check_interval=5, con_num=6,
    )

    print(f"[DEBUG] 提取多边形面积信号 (target_hand={DEBUG_TARGET_HAND})")
    signal = extractor.extract(
        keypoints, DEBUG_VIDEO_PATH,
        target_hand=DEBUG_TARGET_HAND,
        save_csv_path=DEBUG_OUTPUT_CSV,
    )
    print(f"[DEBUG] 共 {len(signal)} 帧, 有效 {int(signal['is_valid'].sum())} 帧")
    print(f"[DEBUG] 信号 CSV: {DEBUG_OUTPUT_CSV}")
# ============================================================
# <<<<<<<<<<<<<<<<<<  END DEBUG / DEV-ONLY  <<<<<<<<<<<<<<<<<<
# ============================================================