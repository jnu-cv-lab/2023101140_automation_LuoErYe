#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // ---------------------------------------------------------
    // 任务1：使用 OpenCV 读取一张测试图片
    // 注意：请确保在 zuoye2 根目录下放了一张名为 test.jpg 的图片
    // ---------------------------------------------------------
    string image_path = "test.jpg";
    Mat img = imread(image_path);

    if (img.empty()) {
        cout << "❌ 错误：无法读取图片！请检查 zuoye2 文件夹下是否有 test.jpg" << endl;
        return -1;
    }

    // ---------------------------------------------------------
    // 任务2：输出图像基本信息
    // ---------------------------------------------------------
    cout << "✅ 成功读取图片！" << endl;
    cout << "--- 图像基本信息 ---" << endl;
    cout << "宽度 (Width): " << img.cols << " 像素" << endl;
    cout << "高度 (Height): " << img.rows << " 像素" << endl;
    cout << "通道数 (Channels): " << img.channels() << endl;
    // 在 C++ OpenCV 中，类型是一个整数代号。例如 16 通常代表 CV_8UC3 (8位无符号3通道)
    cout << "数据类型 (Type ID): " << img.type() << endl; 
    cout << "--------------------" << endl;

    // ---------------------------------------------------------
    // 任务4：转换为灰度图
    // ---------------------------------------------------------
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cout << "✅ 已成功转换为灰度图" << endl;

    // ---------------------------------------------------------
    // 任务5：保存处理结果
    // ---------------------------------------------------------
    imwrite("result_gray.jpg", gray);
    cout << "✅ 灰度图已保存为 result_gray.jpg" << endl;

    // ---------------------------------------------------------
    // 任务6：做一个简单操作（C++ 替代 NumPy 的裁剪操作）
    // 我们裁剪图像的左上角 150x150 区域并保存
    // ---------------------------------------------------------
    // 安全检查：防止原图太小导致裁剪越界报错
    int crop_w = min(150, img.cols);
    int crop_h = min(150, img.rows);
    
    // cv::Rect(x起点, y起点, 宽度, 高度)
    Rect roi(0, 0, crop_w, crop_h); 
    Mat cropped_img = img(roi); // 提取感兴趣区域 (Region of Interest)
    
    imwrite("result_crop.jpg", cropped_img);
    cout << "✅ 裁剪图像已保存为 result_crop.jpg" << endl;

    // ---------------------------------------------------------
    // 任务3：显示原图和灰度图
    // ⚠️ WSL 警告：在 Windows 的 WSL 终端中，弹出图形窗口可能会失败或卡住
    // ---------------------------------------------------------
    cout << "\n正在尝试打开窗口显示图片... (如果没反应或报错，请直接按 Ctrl+C 退出)" << endl;
    try {
        imshow("Original Image", img);
        imshow("Grayscale Image", gray);
        cout << "👉 请在弹出的图片窗口上按任意键退出程序..." << endl;
        waitKey(0); // 等待用户按键盘任意键
    } catch (...) {
        cout << "⚠️ 提示：你的 WSL 环境目前不支持直接弹出 GUI 窗口。" << endl;
        cout << "不过别担心，图片已经成功生成并保存在左侧目录中了！" << endl;
    }

    return 0;
}