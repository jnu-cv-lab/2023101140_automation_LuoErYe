import cv2

# 1. 读取图像
img1 = cv2.imread('/home/albert/cv-course/myproj/zuoye6/box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/albert/cv-course/myproj/zuoye6/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("错误：无法读取图像，请检查图片是否在当前目录下。")
else:
    # 重新提取 ORB 特征 (为了让 task2.py 能独立运行)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. 使用 cv2.BFMatcher() 创建暴力匹配器
    #    ORB 描述子是二进制的，所以使用 cv2.NORM_HAMMING 距离
    #    开启 crossCheck=True 可以互相验证，减少错误匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)

    # 3. 按照匹配距离(distance)从小到大排序
    #    距离越小，说明两个特征点越相似
    matches = sorted(matches, key=lambda x: x.distance)

    # 4. 输出总匹配数量
    total_matches = len(matches)
    print("===== 任务 2：输出信息 =====")
    print(f"总匹配数量: {total_matches}")
    print("============================")

    # 5. 绘制 ORB 初始匹配图 (包含所有匹配)
    #    flags=2 (即 DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 表示不画出没有匹配上的孤立点，让图看起来更干净
    img_matches_all = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imwrite('task2_orb_matches_all.png', img_matches_all)

    # 6. 显示(保存)前 50 个匹配结果
    img_matches_top50 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    cv2.imwrite('task2_orb_matches_top50.png', img_matches_top50)

    print("\n匹配可视化图已成功保存为：")
    print("- task2_orb_matches_all.png (ORB 初始匹配图)")
    print("- task2_orb_matches_top50.png (前50个匹配的可视化结果)")