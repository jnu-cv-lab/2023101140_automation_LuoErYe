import cv2
import numpy as np
import os

# =========================
# 1. 设置工作目录和文件路径
# =========================
WORK_DIR = '/home/albert/cv-course/myproj/zuoye5'
INPUT_IMG_PATH = os.path.join(WORK_DIR, 'A4.jpg')
OUTPUT_IMG_PATH = os.path.join(WORK_DIR, 'corrected_manual.jpg')

# 全局变量，用于存储点击的点和缩放比例
clicked_points = []
scale_ratio = 1.0
img_display = None
original_img = None

# =========================
# 2. 坐标点自动排序函数 (保留的神器)
# =========================
def order_points(pts):
    """
    对四个顶点进行自动排序，顺序为：左上, 右上, 左下, 右下
    这样用户点击时就不需要严格遵守顺序了
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[3] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[2] = pts[np.argmax(diff)]  # 左下
    return rect

# =========================
# 3. 核心透视校正逻辑
# =========================
def do_perspective_transform():
    global original_img, clicked_points, scale_ratio
    print("\n正在计算并进行透视校正...")

    # 1. 将在缩放图上点击的坐标，按比例还原到原图的真实坐标上
    pts_src = np.array(clicked_points, dtype="float32") * scale_ratio
    
    # 2. 自动对四个点进行正确的排序
    rect = order_points(pts_src)
    (tl, tr, bl, br) = rect

    # 3. 计算新图像的动态宽和高，保证画面不被异常拉伸
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]
    ], dtype="float32")

    # 4. 执行透视变换 (针对高清原图进行操作，保证清晰度)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original_img, M, (maxWidth, maxHeight))

    # 5. 保存结果
    cv2.imwrite(OUTPUT_IMG_PATH, warped)
    print(f"✅ 手动校正完成！结果已保存至: {OUTPUT_IMG_PATH}")
    
    # 缩小显示结果，方便查看
    h, w = warped.shape[:2]
    result_display = cv2.resize(warped, (int(w * 800 / h), 800))
    cv2.imshow('Corrected Result', result_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# 4. 鼠标点击回调函数
# =========================
def mouse_click(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"📍 已记录第 {len(clicked_points)} 个点: ({x}, {y})")
        
        # 画一个红色的实心圆标记点击位置
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Click 4 corners of the paper', img_display)

        # 收集满4个点，自动触发变换
        if len(clicked_points) == 4:
            cv2.destroyWindow('Click 4 corners of the paper')
            do_perspective_transform()

# =========================
# 5. 主程序入口
# =========================
if __name__ == "__main__":
    original_img = cv2.imread(INPUT_IMG_PATH)

    if original_img is None:
        print(f"找不到图片，请检查路径: {INPUT_IMG_PATH}")
    else:
        # 为了能在屏幕上完整显示并方便点击，将显示图片等比例缩小到高度 800
        h, w = original_img.shape[:2]
        display_h = 800
        scale_ratio = h / display_h  # 记录缩放比例，后续算坐标要乘回来
        display_w = int(w / scale_ratio)
        
        img_display = cv2.resize(original_img, (display_w, display_h))

        print("=== 操作说明 ===")
        print("请在弹出的图片窗口中，点击A4纸的四个角。")
        print("提示：随意顺序点击即可，程序会自动识别左上、右上、左下、右下。")
        print("================")

        cv2.namedWindow('Click 4 corners of the paper')
        cv2.setMouseCallback('Click 4 corners of the paper', mouse_click)
        cv2.imshow('Click 4 corners of the paper', img_display)
        
        # 等待操作
        cv2.waitKey(0)
        cv2.destroyAllWindows()