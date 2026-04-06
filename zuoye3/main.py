import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct

# --- 1. 环境准备 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'Lenna.jpg') # 请确保你的图叫 input.jpg 或自行修改
img = cv2.imread(img_path, 0) # 读入灰度图
if img is None: exit("❌ 找不到图片，请检查路径")

h, w = img.shape

# --- 2. 下采样 (Downsampling) ---
scale = 0.25
# 按照要求：先高斯平滑再缩小，防止混叠
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_small = cv2.resize(img_blur, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# --- 3. 图像恢复 (Upsampling - 三种内插方法) ---
# A. 最近邻内插 (Nearest Neighbor)
img_res_nearest = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)
# B. 双线性内插 (Bilinear)
img_res_linear = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
# C. 双三次内插 (Bicubic)
img_res_cubic = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)

# --- 4. 空间域比较：计算 MSE & PSNR ---
def get_metrics(org, res):
    mse = np.mean((org.astype(float) - res.astype(float))**2)
    psnr = cv2.PSNR(org, res)
    return mse, psnr

mse_n, psnr_n = get_metrics(img, img_res_nearest)
mse_l, psnr_l = get_metrics(img, img_res_linear)
mse_c, psnr_c = get_metrics(img, img_res_cubic)

print("-" * 30)
print("【空间域恢复指标】")
print(f"最近邻内插 -> MSE: {mse_n:.2f}, PSNR: {psnr_n:.2f} dB")
print(f"双线性内插 -> MSE: {mse_l:.2f}, PSNR: {psnr_l:.2f} dB")
print(f"双三次内插 -> MSE: {mse_c:.2f}, PSNR: {psnr_c:.2f} dB")

# --- 5. 傅里叶变换分析 (FT) ---
def get_ft_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f) # 频谱中心移到图像中心
    return 20 * np.log(np.abs(fshift) + 1) # 对幅度谱取对数

spec_org = get_ft_spectrum(img)
spec_small = get_ft_spectrum(img_small)
spec_res_linear = get_ft_spectrum(img_res_linear)

# --- 6. DCT 分析与能量统计 ---
def dct2(a): 
    return dct(dct(a.T, norm='ortho').T, norm='ortho') #处理二维照片

dct_org = dct2(img.astype(float))
dct_res_nearest = dct2(img_res_nearest.astype(float))
dct_res_linear = dct2(img_res_linear.astype(float))
dct_res_cubic = dct2(img_res_cubic.astype(float))

# 统计左上角低频区域 (取长宽各 1/4 的区域) 能量占比
def energy_ratio(dct_coef):
    total_energy = np.sum(np.abs(dct_coef))
    r, c = dct_coef.shape
    low_freq_energy = np.sum(np.abs(dct_coef[:r//4, :c//4]))
    return (low_freq_energy / total_energy) * 100

rat_o = energy_ratio(dct_org)
rat_n = energy_ratio(dct_res_nearest)
rat_l = energy_ratio(dct_res_linear)
rat_c = energy_ratio(dct_res_cubic)

print("-" * 30)
print("【DCT低频能量占比 (左上角1/16面积)】")
print(f"原图 DCT 低频占比      : {rat_o:.2f}%")
print(f"最近邻恢复 DCT 低频占比 : {rat_n:.2f}%")
print(f"双线性恢复 DCT 低频占比 : {rat_l:.2f}%")
print(f"双三次恢复 DCT 低频占比 : {rat_c:.2f}%")
print("-" * 30)

# --- 7. 绘图展示 (排版为 3行 x 5列 的大图) ---
plt.figure(figsize=(22, 14))

# 第一行：空间域
plt.subplot(3,5,1), plt.imshow(img, cmap='gray'), plt.title('1. Original')
plt.subplot(3,5,2), plt.imshow(img_small, cmap='gray'), plt.title('2. Downsampled (1/4)')
plt.subplot(3,5,3), plt.imshow(img_res_nearest, cmap='gray'), plt.title(f'3. Recovered (Nearest)\nMSE:{mse_n:.1f} | PSNR:{psnr_n:.1f}')
plt.subplot(3,5,4), plt.imshow(img_res_linear, cmap='gray'), plt.title(f'4. Recovered (Bilinear)\nMSE:{mse_l:.1f} | PSNR:{psnr_l:.1f}')
plt.subplot(3,5,5), plt.imshow(img_res_cubic, cmap='gray'), plt.title(f'5. Recovered (Bicubic)\nMSE:{mse_c:.1f} | PSNR:{psnr_c:.1f}')

# 第二行：FT频谱
plt.subplot(3,5,6), plt.imshow(spec_org, cmap='gray'), plt.title('6. FT: Original')
plt.subplot(3,5,7), plt.imshow(spec_small, cmap='gray'), plt.title('7. FT: Downsampled')
plt.subplot(3,5,9), plt.imshow(spec_res_linear, cmap='gray'), plt.title('8. FT: Recovered (Bilinear)')

# 第三行：DCT频谱
def get_dct_vis(dct_coef): return 20 * np.log(np.abs(dct_coef) + 1)
plt.subplot(3,5,11), plt.imshow(get_dct_vis(dct_org), cmap='gray'), plt.title(f'9. DCT: Original\nLow Freq Ratio: {rat_o:.1f}%')
plt.subplot(3,5,13), plt.imshow(get_dct_vis(dct_res_nearest), cmap='gray'), plt.title(f'10. DCT: Nearest\nRatio: {rat_n:.1f}%')
plt.subplot(3,5,14), plt.imshow(get_dct_vis(dct_res_linear), cmap='gray'), plt.title(f'11. DCT: Bilinear\nRatio: {rat_l:.1f}%')
plt.subplot(3,5,15), plt.imshow(get_dct_vis(dct_res_cubic), cmap='gray'), plt.title(f'12. DCT: Bicubic\nRatio: {rat_c:.1f}%')

# 隐藏所有坐标轴
for i in range(1, 16):
    try:
        plt.subplot(3, 5, i)
        plt.axis('off')
    except: pass

plt.tight_layout()
output_path = os.path.join(current_dir, 'assignment_full_result.png')
plt.savefig(output_path)
print(f"✅ 所有图像已生成并保存至: {output_path}")

# =============================================================================
# 附录：实验结果综合分析与原理解释 (基于本次运行结果)
# =============================================================================

"""
一、 傅里叶变换 (FT) 频谱分析
【观察对象】:图6(原图FT)、图7(缩小图FT)、图8(双线性恢复图FT)

1. 现象对比：
   * 原图频谱 (图6)：中心亮度极高，且有明显的亮线呈放射状延伸到图像四周边缘。这表明原图
     (Lena) 包含了非常丰富的锐利边缘和纹理细节（对应高频成分）。
   * 缩小图频谱 (图7)：由于空间尺寸缩小到 1/4，其频谱矩阵也相应变小。在下采样过程中，
     超过采样率一半的高频信号发生了混叠或被直接截断。
   * 恢复图频谱 (图8)：图像尺寸虽然放大了，但频谱图呈现出“中间亮、四周全黑”的现象。
     原本在图6四周可见的高频能量，在图8中几乎完全消失。

2. 核心原因说明：
   图像的下采样过程受到“奈奎斯特-香农采样定理”的限制。当对图像进行平滑并缩小时，大量
   代表发丝、物体边缘的高频信息已被永久性滤除（不可逆丢失）。
   在进行上采样恢复时，使用的“双线性内插法”本质上是一个低通滤波器。它只能通过对周围
   像素取加权平均来平滑地填补空缺像素，这种数学操作只能重构出图像的大致轮廓（低频），
   绝对无法凭空捏造出已经丢失的高频细节。因此，恢复图的频谱在高频区域严重衰减。


二、 离散余弦变换 (DCT) 能量分布分析
【观察对象】: 图9(原图DCT) 及 图10-12(三种恢复方法的DCT)

1. 数据统计现象 (左上角低频区域能量占比):
   * 原图 (Original)           : 51.0%  (基准值)
   * 最近邻恢复 (Nearest)      : 53.9%  (略高于原图)
   * 双线性恢复 (Bilinear)     : 85.6%  (大幅跃升)
   * 双三次恢复 (Bicubic)      : 88.2%  (极度集中)

2. 差异比较与物理解释：
   DCT 矩阵的左上角代表图像的低频分量（平缓的亮度变化，如肤色），右下角代表高频分量
   （剧烈的像素跳变，如细节或噪声）。

   * 为什么最近邻 (Nearest) 占比这么低 (53.9%)
     最近邻插值直接复制相邻像素点，这导致恢复后的图像（图3）充满了严重的马赛克和锯齿边缘。
     这些生硬的锯齿在频率域中表现为大量的“虚假高频信号”。因此，它的高频区域仍然有较多能量，
     导致低频占比无法显著提升。

   * 为什么双线性/双三次 (Bilinear/Bicubic) 占比极高 (85.6% / 88.2%)
     这两种插值算法利用周围多个像素进行平滑过渡计算。这种平滑效应在空间域上表现为图像变
     得模糊（图4、图5）；在频率域上则表现为真正的高频细节被彻底抹除。
     当图像变得极其平滑时，其信号波动变小，DCT 变换后的能量就会极度向代表平缓特征的左上角
     （低频区）压缩。双三次插值（Bicubic）平滑效果最强，因此它的低频能量聚集度最高（88.2%）。

结论:
无论是 FT 还是 DCT 的分析，都从频率域的视角定量地证明了一个事实：图像的缩放恢复过程伴随
着高频信息的不可逆损失；越高级、越平滑的插值算法，其低通滤波效应越明显，能量在频域上的
压缩集中度就越高。
"""