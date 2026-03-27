# OpenCV 基础图像处理项目 (zuoye2)

这是计算机视觉课程的第二次作业，主要展示了如何在 Linux (WSL/Ubuntu) 环境下使用 C++ 和 OpenCV 库进行基础的图像读取、处理与保存操作。

## 1. 项目结构
* `src/`: 存放源代码 (`main.cpp`)
* `build/`: 存放编译生成的二进制可执行文件
* `.vscode/`: 包含项目的编译与调试配置 (`tasks.json`, `launch.json`)
* `test.jpg`: 用于测试的原始图片

## 2. 环境要求
* 操作系统: Ubuntu (WSL)
* 编译器: g++
* 依赖库: libopencv-dev (OpenCV 4.x)

## 3. 运行方法
1. 确保已安装 OpenCV 库。
2. 在 VS Code 中打开 `zuoye2` 文件夹。
3. 切换到 `src/main.cpp` 文件，按下 `F5` 进行编译并运行。
4. 运行结果将保存在项目根目录下（`result_gray.jpg` 和 `result_crop.jpg`）。
