# 任务 1：环境准备
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split, DataLoader

# 检查当前环境是否支持 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# 简单的 Tensor 操作测试
test_tensor = torch.rand(3, 3).to(device)
print(f"PyTorch 测试 Tensor:\n{test_tensor}\n")

# ---------------------------------------------------------

# 任务 2：加载图像数据集
# CIFAR-10 包含彩色图像，需要将其转换为 Tensor 并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集 (50,000 张)
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 下载并加载测试集 (10,000 张)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 将训练集进一步划分为训练集 (40,000) 和验证集 (10,000)
train_size = 40000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# ---------------------------------------------------------

# 任务 3：定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入尺寸: 3 通道, 32x32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # 展平操作
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# ---------------------------------------------------------

# 任务 4 & 5：训练与验证模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10 

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("\n开始训练...")
for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    
    # --- 验证阶段 ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

print("训练完成！")

# ---------------------------------------------------------

# 任务 6：测试模型
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

final_test_loss = test_loss / len(test_loader)
final_test_acc = 100 * correct_test / total_test

print(f"\n测试集最终表现: Loss = {final_test_loss:.4f}, Accuracy = {final_test_acc:.2f}%")



# ---------------------------------------------------------

# 任务 7：绘制训练曲线并保存
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# 绘制 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()

# 保存曲线图为 jpg (彻底去除了 plt.show() 避免终端报错)
plt.savefig('training_curves.jpg')
print("训练曲线已保存为 training_curves.jpg\n所有实验流程执行完毕！请检查当前目录下的 3 张 .jpg 图片。")
plt.close()


# ---------------------------------------------------------
# 任务 5：Feature map (特征图) 可视化
print("\n========== 执行任务 5：第一层 Feature Map 可视化 ==========")

# 1. 从测试集中获取一张图片
dataiter_test = iter(test_loader)
images, labels = next(dataiter_test)
# 取当前 batch 的第一张图片，并增加 batch 维度 (变成 1x3x32x32)
single_image = images[0].unsqueeze(0).to(device) 

# 2. 将图片仅输入到第一层卷积，并经过 ReLU 激活函数获取实际的激活响应
model.eval()
with torch.no_grad():
    conv1_out = model.conv1(single_image)
    fmaps = model.relu1(conv1_out) # 此时形状为: (1, 16, 32, 32)

# 3. 提取前 8 张 feature map 并转移到 CPU 上
# 取出 batch 中的第 0 个元素，以及前 8 个通道，形状变为 (8, 32, 32)
fmaps_8 = fmaps[0, :8, :, :].cpu() 

# 为了使用 save_image 保存为灰度图，我们需要在第 1 维增加一个单通道维度
# 最终形状变为 (8, 1, 32, 32)
fmaps_8 = fmaps_8.unsqueeze(1)

# 4. 保存原始图片（方便对照观察）和特征图
torchvision.utils.save_image(images[0], 'task5_original_image.jpg', normalize=True)
torchvision.utils.save_image(fmaps_8, 'task5_feature_maps.jpg', normalize=True, nrow=8)

print(f"该测试图片的真实类别是: {classes[labels[0]]}")
print("已在当前目录保存原始图片: task5_original_image.jpg")
print("已在当前目录保存 8 张特征图: task5_feature_maps.jpg")