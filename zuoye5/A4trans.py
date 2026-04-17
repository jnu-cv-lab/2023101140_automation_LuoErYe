import cv2
import numpy as np

# 用于存储鼠标点击的坐标
clicked_points = []

def mouse_click(event, x, y, flags, param):
    """鼠标点击事件的回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # 记录左键点击的坐标
        clicked_points.append([x, y])
        print(f"已记录点 {len(clicked_points)}: ({x}, {y})")
        
        # 在图片上画一个红色的圆圈标记，方便你看自己点在了哪里
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Original Image', img_copy)

        # 当集齐4个点时，自动触发透视校正
        if len(clicked_points) == 4:
            print("四个点已收集，正在计算并进行透视校正...")
            do_perspective_transform()

def do_perspective_transform():
    """执行透视变换的核心逻辑"""
    # 1. 整理源坐标 (你刚刚点击的四个点)
    pts_src = np.float32(clicked_points)

    # 2. 设定目标坐标 
    # A4纸的标准比例是 210mm x 297mm。为了保证清晰度，我们将其放大3倍作为像素尺寸
    width, height = 210 * 3, 297 * 3
    
    # 这里的顺序必须和你点击的顺序完全一致！(左上 -> 右上 -> 左下 -> 右下)
    pts_dst = np.float32([
        [0, 0],          # 对应左上角
        [width, 0],      # 对应右上角
        [0, height],     # 对应左下角
        [width, height]  # 对应右下角
    ])

    # 3. 计算透视变换矩阵 (调用了作业允许使用的函数)
    # cv2.getPerspectiveTransform 会根据源点和目标点计算出一个 3x3 的变换矩阵
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # 4. 执行投影变换
    result_img = cv2.warpPerspective(original_img, matrix, (width, height))

    # 5. 显示结果并保存
    cv2.imshow('Corrected Image', result_img)
    cv2.imwrite('corrected_result.jpg', result_img)
    print("校正完成！结果已自动保存为当前目录下的 'corrected_result.jpg'")

# ================= 主程序开始 =================

# 1. 读取图片 (请把 'your_photo.jpg' 换成你拍的照片的实际文件名)
image_path = '/home/albert/cv-course/myproj/zuoye5/A4.jpg' 
original_img = cv2.imread(image_path)

if original_img is None:
    print("错误：找不到图片，请检查图片是否在当前文件夹，以及文件名是否拼写正确！")
else:
    # 手机拍的照片通常非常大，屏幕可能显示不全。
    # 我们将显示用的图片等比例缩小到高度为 800 像素，方便你点击。
    h, w = original_img.shape[:2]
    new_h = 800
    new_w = int((new_h / h) * w)
    original_img = cv2.resize(original_img, (new_w, new_h))

    # 复制一份图片用于画标记，以免破坏原图
    img_copy = original_img.copy()

    # 创建窗口并绑定鼠标点击事件
    cv2.namedWindow('Original Image')
    cv2.setMouseCallback('Original Image', mouse_click)

    print("=== 操作说明 ===")
    print("请按照以下严格的顺序，在弹出的图片上点击A4纸的四个角：")
    print("1. 左上角")
    print("2. 右上角")
    print("3. 左下角")
    print("4. 右下角")
    print("================")

    cv2.imshow('Original Image', img_copy)

    # 等待用户操作。按任意键盘按键，或者关掉窗口就会结束程序
    cv2.waitKey(0)
    cv2.destroyAllWindows()