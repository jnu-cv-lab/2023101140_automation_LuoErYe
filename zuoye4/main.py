import numpy as np
import matplotlib
# 为服务器环境设置非交互式后端，防止报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from scipy.fft import fft2, ifft2, fftshift, ifftshift # 导入完整的 FFT 工具箱
import os

# 创建输出目录
output_dir = 'homework_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_fft_magnitude(image):
    """计算图像的对数傅里叶频谱幅度"""
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    return np.log(np.abs(f_shift) + 1e-8)

# ==========================================
# 第一部分：混叠与傅里叶变换分析
# ==========================================
def run_part1():
    print("正在处理第一部分：生成棋盘格与Chirp图并分析混叠...")
    size = 512
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 1. 生成 Chirp 信号图像
    chirp_img = np.sin(50 * np.pi * (X**2 + Y**2)) 
    
    # 2. 生成 棋盘格 图像 (用于观察方块边缘的混叠)
    # 通过设置较小的频率或直接构造矩阵
    check_size = 32 
    checkerboard = (np.kron([[1, 0] * 8, [0, 1] * 8] * 8, np.ones((check_size, check_size))))
    # 裁剪到 512x512
    checkerboard = checkerboard[:size, :size]
    
    M = 4 # 下采样倍数
    
    # 对 Chirp 进行实验对比
    down_direct = chirp_img[::M, ::M]
    sigma_opt = 1.8 
    filtered = gaussian_filter(chirp_img, sigma=sigma_opt)
    down_filtered = filtered[::M, ::M]

    # 绘制 Chirp 结果图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Part 1: Impact of Filtering on Downsampling (Chirp)', fontsize=16)

    axes[0, 0].imshow(chirp_img, cmap='gray')
    axes[0, 0].set_title('Original Chirp')
    
    axes[0, 1].imshow(down_direct, cmap='gray')
    axes[0, 1].set_title('Direct Downsampling (M=4)\n(Aliasing/Moire)')
    
    axes[0, 2].imshow(down_filtered, cmap='gray')
    axes[0, 2].set_title(f'Gaussian Filtered (sigma={sigma_opt})\n+ Downsampling')

    axes[1, 0].imshow(get_fft_magnitude(chirp_img), cmap='magma')
    axes[1, 0].set_title('Original Spectrum')
    
    axes[1, 1].imshow(get_fft_magnitude(down_direct), cmap='magma')
    axes[1, 1].set_title('Aliased Spectrum')
    
    axes[1, 2].imshow(get_fft_magnitude(down_filtered), cmap='magma')
    axes[1, 2].set_title('Anti-aliased Spectrum')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/part1_chirp_analysis.png')
    
    # 绘制 棋盘格 结果图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Part 1: Downsampling on Checkerboard', fontsize=16)
    
    axes2[0].imshow(checkerboard, cmap='gray')
    axes2[0].set_title('Original Checkerboard')
    
    axes2[1].imshow(checkerboard[::M, ::M], cmap='gray')
    axes2[1].set_title('Direct Downsampling (M=4)')
    
    # 滤波后的棋盘格下采样
    checker_filtered = gaussian_filter(checkerboard, sigma=sigma_opt)
    axes2[2].imshow(checker_filtered[::M, ::M], cmap='gray')
    axes2[2].set_title('Filtered Downsampling')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/part1_checkerboard.png')
    plt.close('all')

def run_part2():
    print("正在处理第二部分：固定 M=4，对比特定 Sigma 并加入原始图对比...")
    size = 512
    M = 4 
    X, Y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # 使用 Chirp 信号，频率随半径增加，容易观察混叠
    chirp_img = np.sin(80 * np.pi * (X**2 + Y**2)) 
    
    # 1. 制作理想参照图 (Ground Truth)
    F = fftshift(fft2(chirp_img))
    radius = size // (2 * M)
    Y_grid, X_grid = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X_grid - size//2)**2 + (Y_grid - size//2)**2)
    ideal_mask = dist_from_center <= radius
    ideal_filtered = np.real(ifft2(ifftshift(F * ideal_mask)))
    ideal_down = ideal_filtered[::M, ::M] # 这是我们希望逼近的极限
    
    # 2. 扫描 MSE 曲线（为了找最优趋势）
    test_sigmas = np.linspace(0.1, 5.0, 100)
    mses = []
    for s in test_sigmas:
        blur = gaussian_filter(chirp_img, sigma=s)
        down = blur[::M, ::M]
        mse = np.mean((down - ideal_down)**2)
        mses.append(mse)
    
    mses = np.array(mses)
    best_sigma = test_sigmas[np.argmin(mses)]
    
    # 3. 准备实验要求的四个特定 Sigma 结果
    target_sigmas = [0.5, 1.0, 2.0, 4.0]
    target_results = [gaussian_filter(chirp_img, sigma=s)[::M, ::M] for s in target_sigmas]
    target_mses = [np.mean((res - ideal_down)**2) for res in target_results]

    # 4. 可视化：3行 (第一行原始对比，第二行曲线，第三行不同Sigma对比)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(r'Part 2: Quantitative Sigma Analysis (Fixed $M=4$)', fontsize=18)
    
    # --- 第一行：原始与理想参照 ---
    ax_orig = plt.subplot(3, 3, 1)
    ax_orig.imshow(chirp_img, cmap='gray')
    ax_orig.set_title('1. Original High-Res Image')
    
    ax_no_filter = plt.subplot(3, 3, 2)
    ax_no_filter.imshow(chirp_img[::M, ::M], cmap='gray')
    ax_no_filter.set_title('2. Direct Downsampling\n(Heavy Aliasing)')
    
    ax_ideal = plt.subplot(3, 3, 3)
    ax_ideal.imshow(ideal_down, cmap='gray')
    ax_ideal.set_title('3. Ideal Low-pass Downsampling\n(Ground Truth)')

    # --- 第二行：MSE 曲线 ---
    ax_curve = plt.subplot(3, 1, 2)
    ax_curve.plot(test_sigmas, mses, 'k-', alpha=0.3)
    colors = ['r', 'g', 'b', 'm']
    for i, s in enumerate(target_sigmas):
        ax_curve.plot(s, target_mses[i], marker='o', color=colors[i], markersize=10, 
                      label=f'Exp $\sigma={s}$: MSE={target_mses[i]:.4f}')
    
    ax_curve.axvline(x=1.8, color='cyan', linestyle='--', label='Theory $\sigma=1.8$')
    ax_curve.set_xlabel('Gaussian $\sigma$')
    ax_curve.set_ylabel('MSE (vs Ideal)')
    ax_curve.legend()
    ax_curve.grid(True)

    # --- 第三行：四个 Sigma 实验图对比 ---
    for i, s in enumerate(target_sigmas):
        ax = plt.subplot(3, 4, 9 + i)
        ax.imshow(target_results[i], cmap='gray')
        ax.set_title(f'$\sigma={s}$')
        ax.axis('off')
        # 在图下方标注状态
        desc = "Aliasing" if s < 1.5 else ("Optimal" if s < 3.0 else "Blurry")
        ax.text(size/(2*M), size/M + 10, desc, ha='center', color=colors[i], fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{output_dir}/part2_comprehensive_analysis.png')
    plt.close()

    print(f"-> 理论最优: 1.8 | 实验最优: {best_sigma:.2f}")

def run_part3():
    print("Processing Part 3: Real Spatially-Adaptive Filtering...")
    size = 512
    M = 4
    X, Y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # 1. 构造混合频率图像 (左侧低频，右侧极高频)
    img = np.zeros((size, size))
    img[:, :size//2] = np.sin(10 * np.pi * X[:, :size//2]) 
    img[:, size//2:] = (np.sin(140 * np.pi * X[:, size//2:]) * np.sin(140 * np.pi * Y[:, size//2:]) > 0).astype(float)
    
    # 2. 估计局部 M 值需求 (通过梯度分析)
    # 梯度越大，代表频率越高，对抗混叠滤波的需求(Sigma)就越大
    grad = np.sqrt(sobel(img, axis=1)**2 + sobel(img, axis=0)**2)
    grad_smoothed = gaussian_filter(grad, sigma=10.0) # 平滑处理以获得区域感
    complexity = grad_smoothed / (np.max(grad_smoothed) + 1e-8) # 归一化到 [0, 1]

    # 3. 真正的空间变异滤波 (对不同区域应用不同 Sigma)
    # 定义映射逻辑：低频区 sigma=0.4 (保锐度), 高频区 sigma=2.0 (理论值防混叠)
    s_min, s_max = 0.4, 2.0
    img_adaptive = np.zeros_like(img)
    
    # 我们采用“多级权重融合”来实现真正的空间变异：
    # 每个像素根据其复杂度，决定从哪个 sigma 滤波图中取值
    num_levels = 10 
    for i in range(num_levels):
        s_val = s_min + (s_max - s_min) * (i / (num_levels - 1))
        filtered_i = gaussian_filter(img, sigma=s_val)
        
        # 计算该 sigma 等级对每个像素的贡献权重
        # 这种方式比简单的左右分块更科学，它能处理图中任何位置出现的局部高频
        target_comp = i / (num_levels - 1)
        # 使用高斯加权函数选择对应的复杂度区域
        mask = np.exp(-((complexity - target_comp)**2) / (2 * (0.1)**2))
        img_adaptive += filtered_i * mask
    
    # 权重归一化（修正能量）
    total_weight = np.zeros_like(img)
    for i in range(num_levels):
        target_comp = i / (num_levels - 1)
        total_weight += np.exp(-((complexity - target_comp)**2) / (2 * (0.1)**2))
    img_adaptive /= (total_weight + 1e-8)

    # 4. 建立“理想抗混叠”基准 (Ideal Ground Truth) 用于计算误差
    F = fftshift(fft2(img))
    r_cutoff = size // (2 * M)
    Y_g, X_g = np.ogrid[:size, :size]
    ideal_lowpass = (X_g - size//2)**2 + (Y_g - size//2)**2 <= r_cutoff**2
    img_ideal = np.real(ifft2(ifftshift(F * ideal_lowpass)))
    down_ideal = img_ideal[::M, ::M]

    # 5. 全图统一滤波对比
    img_uniform = gaussian_filter(img, sigma=1.8)
    down_adaptive = img_adaptive[::M, ::M]
    down_uniform = img_uniform[::M, ::M]

    # 6. 计算真正的误差图 (Error Map)
    # 这里计算的是：(统一滤波结果 - 理想结果) vs (自适应结果 - 理想结果)
    error_uniform = np.abs(down_uniform - down_ideal)
    error_adaptive = np.abs(down_adaptive - down_ideal)
    # 改善图：正值代表自适应更接近理想状态
    improvement = error_uniform - error_adaptive

    # --- 可视化 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('1. Original Mixed Image')
    axes[0, 1].imshow(complexity, cmap='hot'); axes[0, 1].set_title('2. Local Complexity (Sigma Mapping)')
    
    # 误差对比图：展示自适应在哪里发挥了作用
    im = axes[0, 2].imshow(improvement, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('3. Error Improvement (Red = Adaptive Better)')
    plt.colorbar(im, ax=axes[0, 2])

    axes[1, 0].imshow(down_uniform, cmap='gray'); axes[1, 0].set_title('4. Uniform Filter ($\sigma=1.8$)')
    axes[1, 1].imshow(down_adaptive, cmap='gray'); axes[1, 1].set_title('5. Spatially-Adaptive Result')
    
    # 局部放大：对比低频区的清晰度
    zoom_u = down_uniform[10:50, 10:50]
    zoom_a = down_adaptive[10:50, 10:50]
    axes[1, 2].imshow(np.hstack([zoom_u, np.ones((40,2)), zoom_a]), cmap='gray')
    axes[1, 2].set_title('6. Zoom (Left: Uniform | Right: Adaptive)')

    plt.tight_layout()
    # 解决非交互环境问题
    save_path = f'{output_dir}/part3_final_adaptive.png'
    plt.savefig(save_path)
    print(f"Part 3 结果已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    run_part1()
    run_part2()
    run_part3()
    print(f"\n任务完成！结果图片保存在: {os.path.abspath(output_dir)}")