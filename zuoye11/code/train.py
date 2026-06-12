import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. 按照任务书与现有文件配置参数
# ==========================================
DATA_DIR = '/mnt/d/computervisionlab/' 

# 直接对应D 盘中的 4 个文件
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.npy')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.npy')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.npy')

INPUT_DIM = 132          # 每帧骨架特征维度
TARGET_FRAMES = 30       # 每段视频统一帧数
D_MODEL = 128            # Transformer 主维度
NHEAD = 4                # 多头注意力 head 数
NUM_LAYERS = 2           # Transformer Encoder 层数
DIM_FEEDFORWARD = 256    # FFN 中间层维度
NUM_CLASSES = 6          # Kaggle 数据集动作类别数
DROPOUT = 0.1            # 防止过拟合

BATCH_SIZE = 32          # 建议配置
LEARNING_RATE = 1e-3     # Adam 优化器建议学习率
EPOCHS = 30              # 课堂实验建议 20-50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# ==========================================
# 2. 自定义数据集类
# ==========================================
class BadmintonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. 定义 Position Embedding 和 Skeleton Transformer
# ==========================================
class PositionalEncoding(nn.Module):
    """加入时间位置信息"""
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
            d_model=D_MODEL, 
            nhead=NHEAD, 
            dim_feedforward=DIM_FEEDFORWARD, 
            dropout=DROPOUT,
            batch_first=True
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
# 4. 训练与测试主循环
# ==========================================
def main():
    print("正在直接从 Windows D 盘加载已切分的数据集...")
    
    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    label_encoder = LabelEncoder()
    all_y = np.concatenate([y_train, y_test])
    label_encoder.fit(all_y)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"数据加载与对齐成功！")
    print(f"训练集: X_train={X_train.shape}, y_train={y_train_encoded.shape}")
    print(f"测试集: X_test={X_test.shape}, y_test={y_test_encoded.shape}")
    
    train_dataset = BadmintonDataset(X_train, y_train_encoded)
    test_dataset = BadmintonDataset(X_test, y_test_encoded)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SkeletonTransformer().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n开始训练 Skeleton Transformer...")
    best_acc = 0.0
    best_preds = []
    best_labels = []
    
    history_train_loss = []
    history_train_acc = []
    history_test_acc = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        test_acc = test_correct / test_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'transformer_best.pth'))
            best_preds = all_preds
            best_labels = all_labels

    print(f"\n训练完成！最高测试集准确率: {best_acc:.4f}")
    
    class_names = [
        'forehand drive', 
        'forehand lift', 
        'forehand net shot', 
        'forehand clear', 
        'backhand drive', 
        'backhand net shot'
    ]
    
    print("\n最佳模型的 Confusion Matrix:")
    print(confusion_matrix(best_labels, best_preds))
    print("\nClassification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    # =========================================================
    # 5. 自动生成并保存 JPG 评估结果与训练曲线大图的代码
    # =========================================================
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        print("\n正在绘制可视化大图...")
        
        # ---------- 图 1：混淆矩阵与分类报告 ----------
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        
        cm_matrix = confusion_matrix(best_labels, best_preds)
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, 
                    ax=axes[0], cbar=True, annot_kws={"size": 11, "weight": "bold"})
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_xlabel('Predicted Action', fontsize=12, labelpad=10)
        axes[0].set_ylabel('True Action', fontsize=12, labelpad=10)
        axes[0].set_xticklabels(class_names, rotation=30, ha='right')
        
        report_dict = classification_report(best_labels, best_preds, target_names=class_names, output_dict=True)
        report_metrics = {
            'Precision': [report_dict[cls]['precision'] for cls in class_names],
            'Recall': [report_dict[cls]['recall'] for cls in class_names],
            'F1-Score': [report_dict[cls]['f1-score'] for cls in class_names]
        }
        supports = [report_dict[cls]['support'] for cls in class_names]
        df_report = pd.DataFrame(report_metrics, index=class_names)
        
        sns.heatmap(df_report, annot=True, fmt='.2f', cmap='GnBu', ax=axes[1], cbar=True,
                    annot_kws={"size": 11, "weight": "bold"})
        axes[1].set_title('Classification Report Metrics', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xticklabels(df_report.columns, fontsize=11)
        
        for i, support in enumerate(supports):
            axes[1].text(3.1, i + 0.5, f'(Support: {support})', va='center', ha='left', fontsize=10, color='#4a5568', fontstyle='italic')
            
        plt.suptitle('Skeleton Transformer - Evaluation Results', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        try:
            eval_output_path = os.path.join(os.path.dirname(__file__), 'evaluation_results.jpg')
        except NameError:
            eval_output_path = 'evaluation_results.jpg'
        plt.savefig(eval_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"评估矩阵图已保存至: {os.path.abspath(eval_output_path)}")
        
        # ---------- 图 2：训练损失与准确率曲线 ----------
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
        epochs_range = range(1, EPOCHS + 1)
        
        # 绘制 Loss 曲线
        axes2[0].plot(epochs_range, history_train_loss, label='Training Loss', color='#e74c3c', marker='o', markersize=4, linewidth=2)
        axes2[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold', pad=10)
        axes2[0].set_xlabel('Epochs', fontsize=12)
        axes2[0].set_ylabel('Loss', fontsize=12)
        axes2[0].grid(True, linestyle='--', alpha=0.6)
        axes2[0].legend(fontsize=11)
        
        # 绘制 Accuracy 曲线
        axes2[1].plot(epochs_range, history_train_acc, label='Training Accuracy', color='#3498db', marker='o', markersize=4, linewidth=2)
        axes2[1].plot(epochs_range, history_test_acc, label='Testing Accuracy', color='#2ecc71', marker='s', markersize=4, linewidth=2)
        axes2[1].set_title('Training & Testing Accuracy over Epochs', fontsize=14, fontweight='bold', pad=10)
        axes2[1].set_xlabel('Epochs', fontsize=12)
        axes2[1].set_ylabel('Accuracy', fontsize=12)
        axes2[1].grid(True, linestyle='--', alpha=0.6)
        axes2[1].legend(fontsize=11)
        
        plt.suptitle('Model Learning Curves', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        try:
            curve_output_path = os.path.join(os.path.dirname(__file__), 'training_curves.jpg')
        except NameError:
            curve_output_path = 'training_curves.jpg'
        plt.savefig(curve_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线图已保存至: {os.path.abspath(curve_output_path)}")
        
    except Exception as e:
        print(f"绘图失败，请检查是否安装了 matplotlib, seaborn 和 pandas 库。错误原因: {e}")

# ==========================================
# 启动入口
# ==========================================
if __name__ == '__main__':
    main()