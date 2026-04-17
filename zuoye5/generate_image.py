import cv2
import numpy as np

# 创建一张 500x500 的白色背景图
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 1. 画矩形 (红色)
cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), 3)

# 2. 画圆 (绿色)
cv2.circle(img, (350, 100), 60, (0, 255, 0), 3)

# 3. 画平行线 (蓝色)
cv2.line(img, (50, 250), (200, 350), (255, 0, 0), 3)
cv2.line(img, (100, 250), (250, 350), (255, 0, 0), 3)

# 4. 画垂直线 (黑色)
# 线段1: 垂直方向
cv2.line(img, (350, 250), (350, 400), (0, 0, 0), 3)
# 线段2: 水平方向
cv2.line(img, (275, 325), (425, 325), (0, 0, 0), 3)

# 保存并显示图像
cv2.imwrite('test_image.png', img)
cv2.imshow('Original Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()