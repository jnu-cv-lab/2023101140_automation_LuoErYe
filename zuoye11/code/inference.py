import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. 配置参数与路径
# ==========================================
DATA_DIR = '/mnt/d/computervisionlab/'
MODEL_WEIGHTS_PATH = os.path.join(DATA_DIR, 'transformer_best.pth')
MEDIAPIPE_MODEL_PATH = os.path.join(DATA_DIR, 'pose_landmarker_full.task')

# 测试的那个视频的绝对路径
VIDEO_PATH = os.path.join(DATA_DIR, 'demo_video.mp4') 

INPUT_DIM = 132          
TARGET_FRAMES = 30       # 保持和训练时完全一致
D_MODEL = 128            
NHEAD = 4                
NUM_LAYERS = 2           
DIM_FEEDFORWARD = 256    
NUM_CLASSES = 6          
DROPOUT = 0.1            

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = [
    'forehand drive', 
    'forehand lift', 
    'forehand net shot', 
    'forehand clear', 
    'backhand drive', 
    'backhand net shot'
]

# ==========================================
# 2. 还原模型结构 
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class SkeletonTransformer(nn.Module):
    def __init__(self):
        super(SkeletonTransformer, self).__init__()
        self.embedding = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, 
            dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# ==========================================
# 3. 提取单视频骨架特征
# ==========================================
def extract_skeleton_from_video(video_path):
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError("找不到 MediaPipe 模型文件 pose_landmarker_full.task！")
        
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            keypoints = []
            for lm in detection_result.pose_landmarks[0]:
                vis = lm.visibility if lm.visibility is not None else 0.0
                keypoints.extend([lm.x, lm.y, lm.z, vis]) 
            frames_features.append(keypoints)
            
    cap.release()
    
    if len(frames_features) == 0:
        print("警告：视频中未检测到任何人！将返回全零张量。")
        return np.zeros((TARGET_FRAMES, INPUT_DIM)) 
        
    frames_features = np.array(frames_features)
    indices = np.linspace(0, len(frames_features) - 1, TARGET_FRAMES, dtype=int)
    return frames_features[indices]

# ==========================================
# 4. 主推理流程
# ==========================================
def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"找不到测试视频: {VIDEO_PATH}")
        print("请随便找一个测试视频，改名叫 demo_video.mp4 放到 D:\\computervisionlab\\ 目录下。")
        return

    print(f"正在处理测试视频: {os.path.basename(VIDEO_PATH)}")
    
    # 步骤 1: 提取骨架并重采样
    skeleton_seq = extract_skeleton_from_video(VIDEO_PATH) # 形状: (30, 132)
    
    # 转换为模型需要的张量格式，并增加 Batch 维度 (1, 30, 132)
    input_tensor = torch.tensor(skeleton_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 步骤 2: 加载模型和权重
    model = SkeletonTransformer().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval() # 切换到推理模式
    
    # 步骤 3: 模型前向传播
    with torch.no_grad():
        logits = model(input_tensor)
        
        # 步骤 4: Softmax 得到概率
        probabilities = F.softmax(logits, dim=1).squeeze(0) # 去掉 Batch 维度
        
    # 步骤 5: 获取最高概率的类别和置信度
    max_prob, max_idx = torch.max(probabilities, dim=0)
    
    predicted_class = class_names[max_idx.item()]
    confidence = max_prob.item()
    
    # 按照任务书要求的格式输出
    print("\n" + "="*30)
    print("推理输出示例：")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    main()