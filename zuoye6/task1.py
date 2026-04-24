import cv2
import numpy as np

# 1. 读取图像 (以灰度模式读取，特征提取通常在灰度图上进行)
img1_path = '/home/albert/cv-course/myproj/zuoye6/box.png'
img2_path = '/home/albert/cv-course/myproj/zuoye6/box_in_scene.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("错误：无法读取图像，请检查图片是否在当前目录下。")
else:
    # 2. 使用 cv2.ORB_create() 创建 ORB 检测器，并设置 nfeatures=1000
    orb = cv2.ORB_create(nfeatures=1000)

    # 3. 使用 detectAndCompute() 得到关键点(kp)和描述子(des)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 4. 输出两幅图像中的关键点数量和描述子的维度
    print("===== 任务 1：输出信息 =====")
    print(f"box.png 的关键点数量: {len(kp1)}")
    print(f"box_in_scene.png 的关键点数量: {len(kp2)}")
    # des的形状是 (特征点数量, 描述子维度)
    print(f"ORB 描述子的维度: {des1.shape[1]} 字节") 
    print("============================")

    # 5. 使用 cv2.drawKeypoints() 可视化关键点 (用绿色画出)
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

    # 保存特征点可视化图，用于作业提交
    cv2.imwrite('task1_box_keypoints.png', img1_kp)
    cv2.imwrite('task1_box_in_scene_keypoints.png', img2_kp)
    
    print("\n特征点可视化图已成功保存为：")
    print("- task1_box_keypoints.png")
    print("- task1_box_in_scene_keypoints.png")