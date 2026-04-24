import cv2
import numpy as np

# 设定绝对路径
img1_path = '/home/albert/cv-course/myproj/zuoye6/box.png'
img2_path = '/home/albert/cv-course/myproj/zuoye6/box_in_scene.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("错误：无法读取图像，请检查路径。")
else:
    print("===== 任务 6：不同 nfeatures 参数对比结果 =====")
    print(f"{'nfeatures':<10} | {'模板图关键点':<10} | {'场景图关键点':<10} | {'匹配数量':<8} | {'RANSAC内点':<10} | {'内点比例':<8} | {'成功定位'}")
    print("-" * 85)

    # 循环遍历三种不同的特征点上限
    for nfeat in [500, 1000, 2000]:
        orb = cv2.ORB_create(nfeatures=nfeat)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 【新增 1】画出当前 nfeatures 的特征点并保存 (只画 box.png 的作为代表即可)
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
        cv2.imwrite(f'/home/albert/cv-course/myproj/zuoye6/task6_nfeat_{nfeat}_1_keypoints.png', img1_kp)

        # 暴力匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        total_matches = len(matches)

        # 【新增 2】画出当前 nfeatures 的初始所有匹配并保存
        img_all_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        cv2.imwrite(f'/home/albert/cv-course/myproj/zuoye6/task6_nfeat_{nfeat}_2_all_matches.png', img_all_matches)

        inliers_count = 0
        inlier_ratio = 0.0
        is_success = "否"

        if total_matches >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                matchesMask = mask.ravel().tolist()
                inliers_count = np.sum(matchesMask)
                inlier_ratio = inliers_count / total_matches

                if inliers_count > 15:
                    is_success = "是"
                
                # 【保留 3】绘制当前 nfeatures 下的 RANSAC 内点匹配图并保存
                draw_params = dict(matchColor=(0, 255, 0),
                                   singlePointColor=None,
                                   matchesMask=matchesMask, # 掩码：只画内点
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
                cv2.imwrite(f'/home/albert/cv-course/myproj/zuoye6/task6_nfeat_{nfeat}_3_inliers.png', img_inlier_matches)

        # 打印这一行的数据表格
        print(f"{nfeat:<10} | {len(kp1):<15} | {len(kp2):<15} | {total_matches:<12} | {inliers_count:<14} | {inlier_ratio:<12.4f} | {is_success}")
    
    print("-" * 85)
    print("\n提示：图片生成完毕！在你的作业目录下，每个参数(500, 1000, 2000)现在都有3张图：")
    print("- task6_nfeat_xxx_1_keypoints.png (特征点图)")
    print("- task6_nfeat_xxx_2_all_matches.png (初始匹配图)")
    print("- task6_nfeat_xxx_3_inliers.png (RANSAC内点图)")

    """
    ===== 任务 6：不同 nfeatures 参数对比结果 =====
nfeatures  | 模板图关键点     | 场景图关键点     | 匹配数量     | RANSAC内点   | 内点比例     | 成功定位
-------------------------------------------------------------------------------------
500        | 453             | 500             | 148          | 31             | 0.2095       | 是
1000       | 865             | 1000            | 287          | 52             | 0.1812       | 是
2000       | 1589            | 1999            | 511          | 67             | 0.1311       | 是
    """

    """
    1. 比较不同 nfeatures 对匹配数量的影响 
    解答： 随着 nfeatures 参数的增加（从 500 增加到 2000），提取出的特征点总数随之增加，
    这使得暴力匹配阶段寻找对应关系的基数变大。因此，总匹配数量会随着 nfeatures 的增加而显著上升。

    2. 比较不同 nfeatures 对 RANSAC 内点比例的影响 
    解答： 尽管总匹配数量和 RANSAC 内点绝对数量通常会增加，但内点比例（inlier ratio）往往会呈现下降趋势（或波动下降）。
    这是因为 FAST 角点检测会优先提取图像中最显著、对比度最强的高质量特征点；当强制要求提取更多特征点（如 2000 个）时，算法不得不去提取那些对比度较弱、纹理不明显或容易受到背景干扰的“低质量”特征点。
    这些低质量点在匹配时极易产生错配，从而导致引入了大量的外点（Outliers），拉低了整体的内点比例。
    
    3. 说明是否特征点越多，定位效果就一定越好 
    解答： 并不是特征点越多，定位效果就一定越好。 > * 质量胜于数量： 计算单应性矩阵（Homography）理论上只需要 4 对正确的匹配点。如果提取的 500 个特征点具有极高的辨识度和稳定性，已经足够 RANSAC 计算出非常精准的变换模型。
    
    干扰与耗时增加： 盲目增加特征点数量会引入大量抗噪性差的弱特征，这不仅会导致误匹配率（外点）急剧上升，迫使 RANSAC 需要更多的迭代次数才能找到正确模型，严重时甚至可能导致 RANSAC 模型估计失败。
    此外，过多的特征点也会大幅拖慢特征提取和匹配的运算速度。因此，在实际工程中，需要根据图像复杂度和实时性需求寻找 nfeatures 的最佳平衡点。

    """