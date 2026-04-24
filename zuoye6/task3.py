import cv2
import numpy as np

# 1. 使用你提供的绝对路径读取图像
img1_path = '/home/albert/cv-course/myproj/zuoye6/box.png'
img2_path = '/home/albert/cv-course/myproj/zuoye6/box_in_scene.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("错误：无法读取图像，请检查路径。")
else:
    # --- 前置步骤：特征提取与匹配 (同任务1和2) ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    total_matches = len(matches) # 总匹配数量

    # ==========================================
    # 任务 3：RANSAC 剔除错误匹配核心代码
    # ==========================================
    
    # 计算 Homography 至少需要 4 对匹配点
    if total_matches >= 4:
        # 1. 从匹配结果中提取两幅图像中的对应点坐标
        # queryIdx 对应 img1 的关键点索引，trainIdx 对应 img2 的关键点索引
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 2. 使用 cv2.findHomography()，方法选择 cv2.RANSAC，重投影误差阈值设为 5.0
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 3. 将返回的 mask 转换为列表，方便后续计算和画图
        # mask 中值为 1 表示该匹配是内点(inlier)，0 表示是外点(outlier)
        matchesMask = mask.ravel().tolist()

        # 4. 计算内点数量和比例
        inliers_count = np.sum(matchesMask)
        inlier_ratio = inliers_count / total_matches

        # 5. 输出作业要求的指标和矩阵
        print("===== 任务 3：需要提交的数据 =====")
        print(f"总匹配数量: {total_matches}")
        print(f"RANSAC 内点数量: {inliers_count}")
        print(f"内点比例 (inlier_ratio): {inlier_ratio:.4f}")
        print(f"\nHomography 矩阵:\n{H}")
        print("==================================")

        # 6. 根据返回的 mask 显示 RANSAC 后的内点匹配
        # 设置画图参数：只画内点，并且用绿色线条显示
        draw_params = dict(matchColor=(0, 255, 0), # 绿色连线
                           singlePointColor=None,
                           matchesMask=matchesMask, # 传入 mask 掩码，只绘制内点
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        
        # 同样使用绝对路径保存图片，确保它保存在你的作业文件夹里
        save_path = '/home/albert/cv-course/myproj/zuoye6/task3_ransac_matches.png'
        cv2.imwrite(save_path, img_ransac)
        print(f"\nRANSAC 后的匹配图已成功保存为：\n{save_path}")

    else:
        print("匹配点不足 4 对，无法计算 Homography 矩阵！")