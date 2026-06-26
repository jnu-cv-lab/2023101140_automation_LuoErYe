import numpy as np
import cv2
import glob
import os 

# ==========================================
# 1. 配置实验参数
# ==========================================
CHESSBOARD_CORNERS_ROWCOUNT = 9  
CHESSBOARD_CORNERS_COLCOUNT = 6  
SQUARE_SIZE_MM = 23.5            
IMAGE_PATH_PATTERN = '/mnt/d/computervisionlab/chessboard/*.jpg'

# ==========================================
# 2. 准备三维世界坐标系中的点
# ==========================================
objp = np.zeros((CHESSBOARD_CORNERS_ROWCOUNT * CHESSBOARD_CORNERS_COLCOUNT, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_ROWCOUNT, 0:CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

objpoints = [] 
imgpoints = [] 

images = glob.glob(IMAGE_PATH_PATTERN)
if len(images) == 0:
    print(f"标定失败：在 {IMAGE_PATH_PATTERN} 下没有找到图片。请检查路径！")
    exit()

print(f"找到 {len(images)} 张图片，开始检测角点并保存图片...")

# 创建文件夹用来单独存放画了角点的图，防止当前目录太乱
if not os.path.exists('corner_results'):
    os.makedirs('corner_results')

# ==========================================
# 3. 循环遍历图片，检测角点并优化
# ==========================================
imageSize = None 

for iname in images:
    img = cv2.imread(iname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if imageSize is None:
        imageSize = gray.shape[::-1] 

    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), None)

    if ret == True:
        objpoints.append(objp)
        corners_acc = cv2.cornerSubPix(gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_acc)

        cv2.drawChessboardCorners(img, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), corners_acc, ret)
        
        base_name = os.path.basename(iname)  
        save_name = f"corner_results/detected_{base_name}" 
        cv2.imwrite(save_name, img)
        
    else:
        print(f"警告: 在图片 {iname} 中未检测到完整的棋盘格角点，已跳过。")

if len(objpoints) == 0:
    print("所有图片均未能检测到角点。")
    exit()

# ==========================================
# 4. 执行相机标定计算
# ==========================================
print("\n正在计算相机内参和畸变参数...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

print("\n" + "="*40)
print("【标定结果 / Calibration Results】")
print("="*40)
print("1. 相机内参矩阵 K (Camera Matrix):\n", mtx)
print("\n2. 畸变参数 D = [k1, k2, p1, p2, k3]:\n", dist.ravel()) 

# ==========================================
# 5. 计算总体重投影误差
# ==========================================
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"\n3. 总体重投影误差: {mean_error:.4f} 像素")

# ==========================================
# 6. 【关键修改】批量生成所有图片的去畸变对比图
# ==========================================
print("\n正在批量生成去畸变对比图...")

# 创建专门的文件夹存放对比图
if not os.path.exists('compare_results'):
    os.makedirs('compare_results')

for iname in images:
    test_img = cv2.imread(iname)
    if test_img is None:
        continue
    
    h, w = test_img.shape[:2]
    # 进行去畸变处理
    undistorted_img = cv2.undistort(test_img, mtx, dist, None, mtx)

    # 缩小一半拼图
    test_img_small = cv2.resize(test_img, (w // 2, h // 2))
    undistorted_img_small = cv2.resize(undistorted_img, (w // 2, h // 2))
    compare_img = cv2.hconcat([test_img_small, undistorted_img_small])

    # 加字
    cv2.putText(compare_img, 'Original Image', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(compare_img, 'Undistorted Image', (w // 2 + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 保存对比图
    base_name = os.path.basename(iname)
    save_path = f"compare_results/compare_{base_name}"
    cv2.imwrite(save_path, compare_img)

print("所有对比图已生成！请前往 'compare_results' 文件夹挑选效果最明显的一张。")
print("标定程序全部运行完毕！")
cv2.destroyAllWindows()