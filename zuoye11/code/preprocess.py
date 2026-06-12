import os
import glob
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------------------------------
# 1. 初始化新版 Pose Landmarker (指向 D 盘模型文件)
# -----------------------------------------------------
# 使用 Linux 访问 D 盘的绝对路径
model_path = '/mnt/d/computervisionlab/pose_landmarker_full.task'

if not os.path.exists(model_path):
    print(f"找不到模型文件！请确保你已经在 Windows 中下载并放到了 D:\\computervisionlab\\ 下")
    exit()

# 配置基本参数
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # 我们依然按每一帧单独处理
    num_poses=1 # 假设视频中主要只有一个运动员
)
detector = vision.PoseLandmarker.create_from_options(options)

# -----------------------------------------------------
# 2. 视频帧提取函数 (基于新版 Tasks API)
# -----------------------------------------------------
def process_video_to_skeleton(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换为 MediaPipe 专属的 Image 对象
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # 执行推理
        detection_result = detector.detect(mp_image)
        
        # 新版 API 支持检测多人，返回的是一个列表。我们只取第一个人 [0]
        if detection_result.pose_landmarks:
            keypoints = []
            for lm in detection_result.pose_landmarks[0]:
                # 新版 visibility 如果检测不到可能是 None，做一个安全处理
                vis = lm.visibility if lm.visibility is not None else 0.0
                keypoints.extend([lm.x, lm.y, lm.z, vis]) 
            frames_features.append(keypoints)
            
    cap.release()
    
    # 转换为 Numpy 数组并重采样到统一帧数 (30帧)
    if len(frames_features) == 0:
        return np.zeros((target_frames, 132)) 
        
    frames_features = np.array(frames_features)
    indices = np.linspace(0, len(frames_features) - 1, target_frames, dtype=int)
    return frames_features[indices]

# -----------------------------------------------------
# 3. 数据集遍历与保存 (直接读取并保存至 D 盘)
# -----------------------------------------------------
# 指向 D 盘的 batminton 文件夹
base_dir = '/mnt/d/computervisionlab/batminton'  

label_map = {
    "forehand_drive": 0, "forehand_lift": 1,
    "forehand_net_shot": 2, "forehand_clear": 3,
    "backhand_drive": 4, "backhand_net_shot": 5
}

X_data, y_data = [], []

print("开始使用 MediaPipe Tasks API 处理视频...")

for class_name, label in label_map.items():
    folder_path = os.path.join(base_dir, class_name)
    if not os.path.exists(folder_path):
        print(f"找不到文件夹: {folder_path}，请检查文件夹是否存在。")
        continue
        
    video_paths = []
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
        video_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    print(f"正在处理: {class_name}, 共 {len(video_paths)} 个视频")
    
    for video_path in video_paths:
        features = process_video_to_skeleton(video_path, target_frames=30)
        X_data.append(features)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

print(f"\n所有视频处理完毕！总形状: X={X_data.shape}, y={y_data.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# -----------------------------------------------------
# 4. 将生成的 .npy 文件直接保存到 D 盘，保护 C 盘
# -----------------------------------------------------
output_dir = '/mnt/d/computervisionlab/'

np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"数据集保存成功！文件已全部生成在 Windows 的 D:\\computervisionlab\\ 目录下。")