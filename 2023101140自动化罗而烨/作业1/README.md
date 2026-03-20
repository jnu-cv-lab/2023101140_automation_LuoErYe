# 实验一：Python视觉开发环境搭建与图像基本读写

## 1. 实验环境
- 操作系统: Windows 11 + WSL (Ubuntu 22.04)
- 开发工具: VS Code
- Python 环境: 虚拟环境 (.venv-basic)
- 核心库: OpenCV, NumPy, Matplotlib

## 2. 项目结构说明
- `.vscode/`: VS Code 项目配置文件
- `src/main.py`: 实验一源代码
- `test.jpg`: 原始测试图片
- `result_gray.jpg`: 处理生成的灰度图
- `result_crop.jpg`: 使用 NumPy 裁剪的 100x100 局部图
- `requirements.txt`: 环境依赖清单

## 3. 实验内容实现
- **任务1 & 2**: 使用 `cv2.imread` 读取图片，并打印其尺寸（259x194）和通道数（3）。
- **任务3 & 4**: 编写了图像转换逻辑。由于 WSL 终端环境不支持 GUI 弹出窗口，已通过保存文件验证。
- **任务5**: 成功将 BGR 图像转为 Gray 灰度图并保存。
- **任务6**: 利用 NumPy 切片功能 `img[0:100, 0:100]` 完成左上角区域裁剪。

## 4. 运行结果展示
- 图像尺寸: 宽 259, 高 194, 通道数 3
- 数据类型: uint8