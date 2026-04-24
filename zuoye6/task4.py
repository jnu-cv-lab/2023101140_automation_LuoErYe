import cv2
import numpy as np

# 1. 设定图像路径
img1_path = '/home/albert/cv-course/myproj/zuoye6/box.png'
img2_path = '/home/albert/cv-course/myproj/zuoye6/box_in_scene.png'

# 读取图像（场景图读取为彩色，方便画红框）
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)

if img1 is None or img2_color is None:
    print("错误：无法读取图像，请检查路径。")
else:
    # --- 前置步骤：获取单应矩阵 (H) ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 得到 Homography 矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        # ==========================================
        # 任务 4 核心逻辑
        # ==========================================
        
        # 1. 获取 box.png 的四个角点 (左上, 左下, 右下, 右上)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # 2. 使用 cv2.perspectiveTransform() 进行角点投影
        # 将模板图的 4 个角点坐标变换到场景图对应的位置
        dst = cv2.perspectiveTransform(pts, H)

        # 3. 使用 cv2.polylines() 在场景图中画出四边形边框
        # 参数说明：[np.int32(dst)] 为投影后的顶点，True 表示闭合图形，(0, 0, 255) 为红色，线宽为 3
        img_result = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

        # 4. 保存最终目标定位结果
        save_path = '/home/albert/cv-course/myproj/zuoye6/task4_target_location.png'
        cv2.imwrite(save_path, img_result)
        
        print("===== 任务 4：目标定位完成 =====")
        print(f"结果图已保存至: {save_path}")
        print("================================")
    else:
        print("无法计算单应矩阵，定位失败。")