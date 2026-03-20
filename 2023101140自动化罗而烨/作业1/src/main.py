import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys  # <--- 必须导入这个库来接收参数

# 1. 动态获取图片路径（不再写死文件名）
# sys.argv[0] 是脚本名，sys.argv[1] 就是你在 launch.json 里传进来的文件名
if len(sys.argv) > 1:
    img_name = sys.argv[1]
else:
    img_name = 'test.jpg' # 如果没传参数，默认找 test.jpg 兜底

img_path = img_name 

if not os.path.exists(img_path):
    print(f"错误：在当前目录 {os.getcwd()} 下找不到图片 {img_path}")
else:
    img = cv2.imread(img_path)

    # 2. 输出图像基本信息
    h, w, c = img.shape
    print(f"图像尺寸: 宽 {w}, 高 {h}, 通道数 {c}")
    print(f"数据类型: {img.dtype}")

    # 3. 显示原图 (注意 BGR 转 RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. 转换为灰度图并显示
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Matplotlib 同时显示两张图
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
    plt.subplot(122), plt.imshow(img_gray, cmap='gray'), plt.title('Gray')
    plt.show()

    # 5. 保存处理结果
    cv2.imwrite('result_gray.jpg', img_gray)
    print("成功：灰度图已保存为 result_gray.jpg")

    # 6. 使用 NumPy 做简单操作：裁剪左上角 100x100 区域
    crop = img[0:100, 0:100]
    cv2.imwrite('result_crop.jpg', crop)
    print("成功：裁剪区域已保存为 result_crop.jpg")