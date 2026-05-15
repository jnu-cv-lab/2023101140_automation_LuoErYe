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

# CIFAR-10 的 10 个类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 修改后的辅助函数：支持将标签作为标题直接画在 jpg 图片上并保存
def imshow(img, filename="sample.jpg", title=None):
    img = img / 2 + 0.5     # 反归一化
    npimg = img.numpy()
    
    plt.figure(figsize=(12, 4)) # 把画布拉宽一点，方便文字对齐图片
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    
    if title:
        # 将文字标题加在图片正上方，使用等宽字体对齐
        plt.title(title, fontsize=14, fontfamily='monospace', loc='left')
        
    plt.savefig(filename, bbox_inches='tight') # 保存图片为 jpg
    plt.close() # 关闭画布防止重叠

# 获取一批训练数据并保存图片
dataiter = iter(train_loader)
images, labels = next(dataiter)

print("正在保存训练集样本图片 (train_samples.jpg)...")
# 提取这 8 张图的真实标签，拼成一个字符串
true_labels_str = '   '.join(f'{classes[labels[j]]:5s}' for j in range(8))
title_str = f"True Labels: {true_labels_str}"
imshow(torchvision.utils.make_grid(images[:8]), filename="train_samples.jpg", title=title_str)

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

# 获取测试集图像并保存预测结果图片
dataiter_test = iter(test_loader)
images, labels = next(dataiter_test)
images_device, labels_device = images.to(device), labels.to(device)

outputs = model(images_device)
_, predicted = torch.max(outputs, 1)

print("\n正在保存测试集预测结果图片 (test_predictions.jpg)...")
# 分别提取真实标签和预测标签
true_labels_str = '   '.join(f'{classes[labels[j]]:5s}' for j in range(8))
pred_labels_str = '   '.join(f'{classes[predicted[j]]:5s}' for j in range(8))
# 用换行符 \n 把两行文字拼在一起
title_str = f"True: {true_labels_str}\nPred: {pred_labels_str}"
imshow(torchvision.utils.make_grid(images[:8]), filename="test_predictions.jpg", title=title_str)

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