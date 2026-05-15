import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np

# ================= 1. 环境与配置 =================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"========== 当前使用计算设备: {device} ==========\n")
EPOCHS = 5
BATCH_SIZE = 64

# ================= 2. 数据加载模块 =================
def get_dataloaders(dataset_name):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Pad(2), # 28x28 填充到 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        in_channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        in_channels = 3
    else:
        raise ValueError("不支持的数据集")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, in_channels

# ================= 3. 模型定义模块 =================
class BaseCNN(nn.Module):
    def __init__(self, in_channels):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    def __init__(self, in_channels):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ================= 4. 训练与测试统配函数 =================
def train_and_evaluate(dataset_name, model_class, optimizer_name, lr=0.001):
    print(f"\n---> 开始实验: {dataset_name} | 模型: {model_class.__name__} | 优化器: {optimizer_name}")
    train_loader, test_loader, in_channels = get_dataloaders(dataset_name)
    
    model = model_class(in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 用于记录绘图数据
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # 测试阶段 (每轮结束都测一次，以便画曲线)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = 100 * correct / total
        history['test_acc'].append(epoch_acc)
        
        print(f"  Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")
        
    return history

# ================= 5. 画图辅助函数 =================
def plot_comparisons(res_mnist, res_cifar_base, res_adv, res_sgd):
    epochs_range = range(1, EPOCHS + 1)
    
    # 1. 优化器对比：Train Loss (Adam vs SGD)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, res_cifar_base['train_loss'], label='Adam Optimizer', marker='o')
    plt.plot(epochs_range, res_sgd['train_loss'], label='SGD Optimizer', marker='x')
    plt.title('Optimizer Comparison: Training Loss (CIFAR-10)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('compare_optimizer_loss.jpg')
    plt.close()

    # 2. 网络结构对比：Test Accuracy (BaseCNN vs AdvancedCNN)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, res_cifar_base['test_acc'], label='BaseCNN', marker='o')
    plt.plot(epochs_range, res_adv['test_acc'], label='AdvancedCNN (Deeper + Dropout)', marker='s')
    plt.title('Architecture Comparison: Test Accuracy (CIFAR-10)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('compare_architecture_acc.jpg')
    plt.close()

    # 3. 综合性能对比：最终准确率柱状图
    plt.figure(figsize=(10, 6))
    labels = ['MNIST (Base+Adam)', 'CIFAR-10 (Base+SGD)', 'CIFAR-10 (Base+Adam)', 'CIFAR-10 (Adv+Adam)']
    final_accs = [
        res_mnist['test_acc'][-1], 
        res_sgd['test_acc'][-1], 
        res_cifar_base['test_acc'][-1], 
        res_adv['test_acc'][-1]
    ]
    
    bars = plt.bar(labels, final_accs, color=['#4CAF50', '#F44336', '#2196F3', '#9C27B0'])
    plt.title('Final Test Accuracy Comparison Across All Experiments')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105) # 给顶部留点空间写数字
    
    # 在柱子顶部加上具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('compare_final_accuracy_bar.jpg')
    plt.close()
    
    print("\n✅ 对比图表已生成：")
    print(" 1. compare_optimizer_loss.jpg (优化器 Loss 曲线)")
    print(" 2. compare_architecture_acc.jpg (网络结构准确率曲线)")
    print(" 3. compare_final_accuracy_bar.jpg (最终准确率柱状图)")

# ================= 6. 主执行逻辑与报告生成 =================
if __name__ == '__main__':
    start_time = time.time()
    
    # 实验 1：MNIST 基础对比
    res_mnist = train_and_evaluate('MNIST', BaseCNN, 'Adam', lr=0.001)
    
    # 实验 2：CIFAR-10 基础基准
    res_cifar_base = train_and_evaluate('CIFAR10', BaseCNN, 'Adam', lr=0.001)
    
    # 实验 3：进阶任务 1 - CIFAR-10 + 高级网络
    res_adv = train_and_evaluate('CIFAR10', AdvancedCNN, 'Adam', lr=0.001)
    
    # 实验 4：进阶任务 2 - CIFAR-10 + SGD对比
    res_sgd = train_and_evaluate('CIFAR10', BaseCNN, 'SGD', lr=0.001)
    
    # 生成图表
    plot_comparisons(res_mnist, res_cifar_base, res_adv, res_sgd)
    
    print("\n" + "="*50)
    print("🎉 所有实验跑完！耗时: {:.1f} 秒".format(time.time() - start_time))
    print("="*50)
    
    # 打印最终表格供报告使用
    print("\n请将以下数据填入你的实验报告：\n")
    
    print("【优化器比较记录表】")
    print("Optimizer\tLearning Rate\tTest Accuracy")
    print(f"SGD\t\t0.001\t\t{res_sgd['test_acc'][-1]:.2f}%")
    print(f"Adam\t\t0.001\t\t{res_cifar_base['test_acc'][-1]:.2f}%")
    print("-" * 40)
    
    print("\n【MNIST 与 CIFAR-10 比较记录表】")
    print("数据集\t\t图像类型\t类别数\t测试准确率\t难度")
    print(f"MNIST\t\t灰度手写数字\t10\t{res_mnist['test_acc'][-1]:.2f}%\t\t低")
    print(f"CIFAR-10\t彩色自然图像\t10\t{res_cifar_base['test_acc'][-1]:.2f}%\t\t高")
    print("-" * 40)
    
    print("\n【网络结构改进效果评价】")
    print(f"基础 CNN 准确率: {res_cifar_base['test_acc'][-1]:.2f}%")
    print(f"进阶 CNN (加深+Dropout) 准确率: {res_adv['test_acc'][-1]:.2f}%")