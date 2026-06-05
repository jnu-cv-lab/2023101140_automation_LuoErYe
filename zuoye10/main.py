import torch
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. 实现 Sinusoidal Position Encoding 
# ==========================================
def get_sinusoidal_position_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ==========================================
# 2. 实现二维向量旋转 
# ==========================================
def rotate_2d(x, theta):
    x1, x2 = x[..., 0], x[..., 1]
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    out_x1 = x1 * cos_theta - x2 * sin_theta
    out_x2 = x1 * sin_theta + x2 * cos_theta
    return torch.stack([out_x1, out_x2], dim=-1)

# ==========================================
# 3. 实现高维 RoPE 
# ==========================================
def apply_rotary_position_embeddings(x, seq_len):
    d_model = x.shape[-1]
    theta_base = 10000.0 ** (-2 * torch.arange(0, d_model // 2).float() / d_model)
    m = torch.arange(seq_len).float()
    freqs = m.unsqueeze(1) * theta_base.unsqueeze(0)
    x_reshaped = x.view(seq_len, d_model // 2, 2)
    x_rotated = rotate_2d(x_reshaped, freqs)
    return x_rotated.view(seq_len, d_model)

# ==========================================
# 4. 绘图函数 (原有的 3 个)
# ==========================================
def plot_sinusoidal_heatmap():
    pe = get_sinusoidal_position_encoding(100, 128)
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy(), cmap='RdBu', aspect='auto') 
    plt.colorbar(label='Encoding Value')
    plt.title("Visualizing Sinusoidal Position Encoding (E+pos)")
    plt.xlabel("Embedding Dimension (d_model)")
    plt.ylabel("Sequence Position (pos)")
    plt.tight_layout()
    plt.savefig("1_Sinusoidal_PE_Heatmap.jpg", dpi=300, format='jpg')
    plt.close()

def plot_rope_2d_rotation():
    seq_len = 16
    base_vector = torch.tensor([[1.0, 0.0]]).expand(seq_len, -1)
    rotated_vectors = apply_rotary_position_embeddings(base_vector, seq_len)
    plt.figure(figsize=(8, 8))
    colors = plt.cm.viridis(torch.linspace(0, 1, seq_len).numpy())
    for i in range(seq_len):
        x, y = rotated_vectors[i, 0].item(), rotated_vectors[i, 1].item()
        plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                  length_includes_head=True, color=colors[i], alpha=0.8)
        plt.text(x * 1.05, y * 1.05, f"m={i}", fontsize=9, color=colors[i])
    plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
    plt.axhline(0, color='grey', lw=0.5, linestyle='--')
    plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.title("RoPE in 2D: Vector Rotation across Positions")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("2_RoPE_2D_Rotation.jpg", dpi=300, format='jpg')
    plt.close()

def plot_rope_relative_decay():
    seq_len, d_model = 100, 64
    q_base = torch.ones(1, d_model)
    k_base = torch.ones(1, d_model)
    q_rope = apply_rotary_position_embeddings(q_base.expand(seq_len, -1), seq_len)
    k_rope = apply_rotary_position_embeddings(k_base.expand(seq_len, -1), seq_len)
    
    dot_products = [torch.dot(q_rope[0], k_rope[i]).item() / d_model for i in range(seq_len)]
    plt.figure(figsize=(10, 5))
    plt.plot(range(seq_len), dot_products, marker='o', markersize=3, color='b')
    plt.title("Attention Score Decay with Relative Distance (RoPE)")
    plt.xlabel("Relative Distance (|m - n|)")
    plt.ylabel("Normalized Dot Product Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("3_RoPE_Relative_Decay.jpg", dpi=300, format='jpg')
    plt.close()

# ==========================================
# 5. 绘图函数 (新增的 3 个对比图)
# ==========================================
def plot_rope_heatmap():
    # 观察 RoPE 作用在均匀向量上的表现
    seq_len, d_model = 100, 128
    base_vector = torch.ones(seq_len, d_model)
    rope_pe = apply_rotary_position_embeddings(base_vector, seq_len)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(rope_pe.numpy(), cmap='RdBu', aspect='auto') 
    plt.colorbar(label='Rotated Value')
    plt.title("Visualizing RoPE (Applied to All-Ones Vector)")
    plt.xlabel("Embedding Dimension (d_model)")
    plt.ylabel("Sequence Position (pos)")
    plt.tight_layout()
    plt.savefig("4_RoPE_Heatmap.jpg", dpi=300, format='jpg')
    plt.close()

def plot_epos_relative_decay():
    # 绘制纯正弦编码 E+pos 自身的点积随距离的变化
    seq_len, d_model = 100, 64
    pe = get_sinusoidal_position_encoding(seq_len, d_model)
    
    dot_products = [torch.dot(pe[0], pe[i]).item() / d_model for i in range(seq_len)]
    plt.figure(figsize=(10, 5))
    plt.plot(range(seq_len), dot_products, marker='x', markersize=3, color='r')
    plt.title("Attention Score Decay with Relative Distance (Pure E+pos)")
    plt.xlabel("Relative Distance (|m - n|)")
    plt.ylabel("Normalized Dot Product Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("5_Epos_Relative_Decay.jpg", dpi=300, format='jpg')
    plt.close()

def plot_translation_invariance_comparison():
    """
    终极对比：固定相对距离为 5，让 Q 和 K 在序列中整体向后平移。
    看 Attention Score 是否会因为绝对位置的改变而波动。
    """
    seq_len, d_model = 50, 64
    relative_dist = 5
    
    # 相同的基础内容向量
    q_base = torch.randn(1, d_model).expand(seq_len, -1)
    k_base = torch.randn(1, d_model).expand(seq_len, -1)
    
    # RoPE 处理
    q_rope = apply_rotary_position_embeddings(q_base, seq_len)
    k_rope = apply_rotary_position_embeddings(k_base, seq_len)
    
    # E+pos 处理 (内容 + 位置)
    pe = get_sinusoidal_position_encoding(seq_len, d_model)
    q_epos = q_base + pe
    k_epos = k_base + pe
    
    rope_scores, epos_scores = [], []
    shift_range = range(0, seq_len - relative_dist)
    
    for shift in shift_range:
        pos_q, pos_k = shift, shift + relative_dist
        rope_scores.append(torch.dot(q_rope[pos_q], k_rope[pos_k]).item())
        epos_scores.append(torch.dot(q_epos[pos_q], k_epos[pos_k]).item())
        
    plt.figure(figsize=(10, 5))
    plt.plot(shift_range, rope_scores, label='RoPE (Constant relative distance = 5)', color='b', linewidth=3)
    plt.plot(shift_range, epos_scores, label='E+pos (Constant relative distance = 5)', color='r', linestyle='--')
    
    plt.title("Translation Invariance: RoPE vs E+pos")
    plt.xlabel("Absolute Position Shift (m)")
    plt.ylabel("Attention Score (Dot Product)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("6_Comparison_Translation_Invariance.jpg", dpi=300, format='jpg')
    plt.close()

# ==========================================
# 用数值实验验证 RoPE 的相对位置性质
# ==========================================
def verify_rope_relative_property():
    print("--- 验证 RoPE 的相对位置性质 ---")
    d_model = 64
    seq_len = 10
    
    # 固定随机种子，保证每次运行生成的数值一样
    torch.manual_seed(42) 
    
    q_base = torch.randn(1, d_model)
    k_base = torch.randn(1, d_model)
    
    q_seq = q_base.expand(seq_len, -1)
    k_seq = k_base.expand(seq_len, -1)
    
    q_rope = apply_rotary_position_embeddings(q_seq, seq_len)
    k_rope = apply_rotary_position_embeddings(k_seq, seq_len)
    
    pos_q1, pos_k1 = 2, 5
    dot_product_1 = torch.dot(q_rope[pos_q1], k_rope[pos_k1])
    
    pos_q2, pos_k2 = 6, 9
    dot_product_2 = torch.dot(q_rope[pos_q2], k_rope[pos_k2])
    
    pos_q3, pos_k3 = 1, 2
    dot_product_3 = torch.dot(q_rope[pos_q3], k_rope[pos_k3])
    
    print(f"位置 {pos_q1} 的 Q 和 位置 {pos_k1} 的 K 的点积 (相对距离为-3): {dot_product_1.item():.4f}")
    print(f"位置 {pos_q2} 的 Q 和 位置 {pos_k2} 的 K 的点积 (相对距离为-3): {dot_product_2.item():.4f}")
    print(f"位置 {pos_q3} 的 Q 和 位置 {pos_k3} 的 K 的点积 (相对距离为-1): {dot_product_3.item():.4f}")
    
    diff = abs(dot_product_1.item() - dot_product_2.item())
    print(f"距离相同的两组点积差值: {diff:.6e}")
    if diff < 1e-5:
        print("=> 验证成功！只要相对距离相同，RoPE 处理后的内积结果就相同。")

if __name__ == "__main__":
    verify_rope_relative_property()
    print("开始生成所有 6 种可视化实验图表...")
    plot_sinusoidal_heatmap()
    plot_rope_2d_rotation()
    plot_rope_relative_decay()
    plot_rope_heatmap()
    plot_epos_relative_decay()
    plot_translation_invariance_comparison()
