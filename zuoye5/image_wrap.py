import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 设置工作目录和参数 (保持不变)
# ==========================================
# 定义工作目录
work_dir = '/home/albert/cv-course/myproj/zuoye5'
img_path = os.path.join(work_dir, 'test_image.png')

# 读取原图
img = cv2.imread(img_path)
if img is None:
    print(f"错误: 找不到文件 {img_path}，请确保上一步已成功生成该图！")
    exit()

rows, cols = img.shape[:2]

# ==========================================
# 2. 执行所有变换 (与你提供的一致)
# ==========================================

# --- 2.1. 相似变换 (Similarity Transform) ---
center = (cols / 2, rows / 2)
angle = 30
scale = 0.8
M_sim = cv2.getRotationMatrix2D(center, angle, scale)
sim_img = cv2.warpAffine(img, M_sim, (cols, rows))
print("相似变换结果已生成。")

# --- 2.2. 仿射变换 (Affine Transform) ---
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[50, 50], [180, 50], [100, 160]])  # 修复了超出边界的问题
M_aff = cv2.getAffineTransform(pts1, pts2)
aff_img = cv2.warpAffine(img, M_aff, (cols, rows))
print("仿射变换结果已生成。")

# --- 2.3. 透视变换 (Perspective Transform) ---
pts_src = np.float32([
    [0, 0], [cols - 1, 0], 
    [0, rows - 1], [cols - 1, rows - 1]
])

# 【修正后的目标点】模拟“侧面看墙”的视角
# 左边保持较高，右边严重压缩（高度从500被挤压到只有140），制造强烈的横向纵深
pts_dst = np.float32([
    [150, 50],              # 左上
    [cols - 20, 200],       # 右上（大幅压缩）
    [150, rows - 50],       # 左下
    [cols - 20, rows - 200] # 右下（形成明显收敛）
])
M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
persp_img = cv2.warpPerspective(img, M_persp, (cols, rows))
print("透视变换结果已生成。")

# ==========================================
# 3. 图像整合与可视化 (新增功能)
# ==========================================

# --- 3.1. 图像预处理 (缩放与添加标题) ---
# 为了让整合后的图片大小一致，我们将所有图片缩放到与原图相同的大小
# (在这个例子中它们已经是相同大小的，但加上这一步更通用)
resize_shape = (cols, rows)
orig_resized = cv2.resize(img, resize_shape)
sim_resized = cv2.resize(sim_img, resize_shape)
aff_resized = cv2.resize(aff_img, resize_shape)
persp_resized = cv2.resize(persp_img, resize_shape)

# 在每个图像上方添加黑色背景的标题，方便区分
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255) # 黑色背景，用白色文字
thickness = 2
line_type = cv2.LINE_AA
title_height = 50

# 辅助函数，用于在图片上方添加标题
def add_title(target_img, text):
    h, w, c = target_img.shape
    new_h = h + title_height
    # 创建带黑边的画布
    canvas = np.zeros((new_h, w, c), dtype=np.uint8)
    canvas[title_height:new_h, 0:w] = target_img
    
    # 放置文字
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (title_height + text_size[1]) // 2
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
    return canvas

# 给每个图加上标题
img_orig_with_title = add_title(orig_resized, "Original")
img_sim_with_title = add_title(sim_resized, "Similarity")
img_aff_with_title = add_title(aff_resized, "Affine")
img_persp_with_title = add_title(persp_resized, "Perspective")

# --- 3.2. 创建拼图 (Composite) ---
# 使用 np.vstack (垂直拼接) 和 np.hstack (水平拼接) 构建 2x2 拼图

# 顶部一行：Original + Similarity
top_row = np.hstack((img_orig_with_title, img_sim_with_title))

# 底部一行：Affine + Perspective
bottom_row = np.hstack((img_aff_with_title, img_persp_with_title))

# 最终整合
composite_img = np.vstack((top_row, bottom_row))

# --- 3.3. 保存整合后的图像 ---
composite_save_path = os.path.join(work_dir, 'composite_result.png')
cv2.imwrite(composite_save_path, composite_img)
print(f"\n整合后的 $2 \\times 2$ 图像已成功保存至: {composite_save_path}")

# --- 3.4. 使用 Matplotlib 显示拼图 (在 Notebook/GUI 环境中查看) ---
# OpenCV 读取的是 BGR 格式，Matplotlib 需要 RGB 格式
composite_img_rgb = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(composite_img_rgb)
plt.title("2x2 Composite Visualization (Original, Sim, Aff, Persp)")
plt.axis("off") # 关闭坐标轴
plt.show()