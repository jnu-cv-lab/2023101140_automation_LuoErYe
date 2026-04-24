import cv2
import numpy as np
import time

# 设定绝对路径
img1_path = '/home/albert/cv-course/myproj/zuoye6/box.png'
img2_path = '/home/albert/cv-course/myproj/zuoye6/box_in_scene.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    print("错误：无法读取图像，请检查路径。")
else:
    print("===== 开始执行 ORB 与 SIFT 对比实验 =====\n")

    # ==========================================
    # 1. ORB 特征匹配 (nfeatures=1000)
    # ==========================================
    start_time = time.time()
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
    
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(des1_orb, des2_orb)
    
    orb_matches_count = len(matches_orb)
    orb_inliers = 0
    orb_ratio = 0.0
    orb_success = "否"
    
    if orb_matches_count >= 4:
        src_pts = np.float32([kp1_orb[m.queryIdx].pt for m in matches_orb]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2_orb[m.trainIdx].pt for m in matches_orb]).reshape(-1, 1, 2)
        H_orb, mask_orb = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H_orb is not None:
            orb_inliers = np.sum(mask_orb)
            orb_ratio = orb_inliers / orb_matches_count
            if orb_inliers > 10: orb_success = "是"
            
    orb_time = time.time() - start_time
    orb_speed_eval = f"{orb_time:.3f} 秒 (极快)"

    # ==========================================
    # 2. SIFT 特征匹配
    # ==========================================
    start_time = time.time()
    
    # 使用 cv2.SIFT_create()
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(img2, None)
    
    # SIFT 描述子是浮点数，必须使用 cv2.NORM_L2
    bf_sift = cv2.BFMatcher(cv2.NORM_L2)
    # 使用 KNN matching (k=2 表示找最近的2个点)
    matches_knn = bf_sift.knnMatch(des1_sift, des2_sift, k=2)
    
    # 使用 Lowe ratio test 筛选匹配
    good_matches = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    sift_matches_count = len(good_matches)
    sift_inliers = 0
    sift_ratio = 0.0
    sift_success = "否"
    
    if sift_matches_count >= 4:
        src_pts = np.float32([kp1_sift[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2_sift[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H_sift, mask_sift = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H_sift is not None:
            sift_inliers = np.sum(mask_sift)
            sift_ratio = sift_inliers / sift_matches_count
            if sift_inliers > 10: sift_success = "是"
            
            # 【新增】画出 SIFT 的 RANSAC 内点匹配图并保存
            draw_params_sift = dict(matchColor=(0, 255, 0), # 绿色连线
                                    singlePointColor=None,
                                    matchesMask=mask_sift.ravel().tolist(), # 只画内点
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_sift_matches = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, good_matches, None, **draw_params_sift)
            cv2.imwrite('/home/albert/cv-course/myproj/zuoye6/task_elective_sift_matches.png', img_sift_matches)

            # 画出 SIFT 定位图并保存
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H_sift)
            img_sift_result = cv2.polylines(img2_color.copy(), [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA) # 用蓝色框表示 SIFT
            cv2.imwrite('/home/albert/cv-course/myproj/zuoye6/task_elective_sift_location.png', img_sift_result)
            
    sift_time = time.time() - start_time
    sift_speed_eval = f"{sift_time:.3f} 秒 (较慢)"

    # ==========================================
    # 3. 打印对比表格
    # ==========================================
    print("-" * 80)
    print(f"{'方法':<10} | {'匹配数量':<10} | {'RANSAC内点数':<15} | {'内点比例':<10} | {'成功定位':<10} | {'运行速度主观评价'}")
    print("-" * 80)
    print(f"{'ORB':<12} | {orb_matches_count:<14} | {orb_inliers:<19} | {orb_ratio:<14.4f} | {orb_success:<14} | {orb_speed_eval}")
    print(f"{'SIFT':<12} | {sift_matches_count:<14} | {sift_inliers:<19} | {sift_ratio:<14.4f} | {sift_success:<14} | {sift_speed_eval}")
    print("-" * 80)
    print("\n注：已保存以下图片：")
    print("1. task_elective_sift_matches.png (SIFT 内点匹配图)")
    print("2. task_elective_sift_location.png (SIFT 目标定位图)")

    """
    
--------------------------------------------------------------------------------
方法         | 匹配数量       | RANSAC内点数       | 内点比例       | 成功定位       | 运行速度主观评价
--------------------------------------------------------------------------------
ORB          | 287            | 52                  | 0.1812         | 是              | 0.053 秒 (极快)
SIFT         | 80             | 75                  | 0.9375         | 是              | 0.074 秒 (较慢)
--------------------------------------------------------------------------------

    """