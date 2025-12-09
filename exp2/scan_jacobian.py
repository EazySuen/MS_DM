import os
import sys

# === 0. 必须最先执行：设置 HF 镜像 & 路径 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 获取当前脚本所在的目录 (绝对路径)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 打印一下路径确认
print(f"📂 Working Directory: {CURRENT_DIR}")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. 配置 (Config)
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 显存敏感，Batch Size 调小一点 (因为要算二阶导/梯度)
    batch_size = 16  
    
    # 扫描样本量 (测 64 张图的平均值足够了)
    num_samples = 64 
    
    # 扫描间隔 (每隔 10 步测一次)
    scan_interval = 10 
    
    # === 路径配置 (全部基于 CURRENT_DIR) ===
    data_root = os.path.join(CURRENT_DIR, "data")
    save_dir = os.path.join(CURRENT_DIR, "analysis_results")

# 初始化目录
os.makedirs(Config.data_root, exist_ok=True)
os.makedirs(Config.save_dir, exist_ok=True)

# ==========================================
# 2. 核心逻辑：刚性计算
# ==========================================
def compute_jacobian_norm(model, x, t):
    """
    使用 Hutchinson 方法估算 Jacobian Frobenius Norm
    衡量系统在当前状态下的"刚性" (Stiffness)
    ||J||_F ≈ E[ || v^T * J || ]
    """
    # 必须开启梯度记录，否则无法对 x 求导
    x = x.detach().clone().requires_grad_(True)
    
    # 1. 模型前向 (Teacher 预测噪声)
    # output shape: [B, 3, 32, 32]
    output = model(x, t).sample
    
    # 2. 随机投影向量 (Rademacher or Gaussian)
    v = torch.randn_like(output)
    
    # 3. 计算 Vector-Jacobian Product (VJP)
    # 这相当于计算 v^T * J，不需要显式构建巨大的 Jacobian 矩阵
    # create_graph=False: 我们只需要数值，不需要对这个 Norm 再求导
    g = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=v,
        create_graph=False, 
        retain_graph=False,
        only_inputs=True
    )[0]
    
    # 4. 范数估算
    # ||J||^2 approx ||g||^2
    # 我们计算每个样本的范数，然后取平均
    # g shape: [B, 3, 32, 32] -> flatten -> [B, D]
    g_flat = g.view(g.shape[0], -1)
    j_norm = torch.norm(g_flat, dim=1).mean().item()
    
    return j_norm

# ==========================================
# 3. 主程序
# ==========================================
def run_stiffness_scan():
    print(f"🚀 Scanning Teacher Stiffness (Jacobian Norm) on {Config.device}...")
    
    # 1. Load Model
    print(">>> Loading Google DDPM...")
    try:
        unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(Config.device)
        scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("请检查网络连接或 HF_ENDPOINT 设置。")
        return

    unet.eval()
    
    # 2. Load Data
    print(f">>> Loading Data from {Config.data_root}...")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 自动下载到相对路径下的 data 文件夹
    dataset = datasets.CIFAR10(root=Config.data_root, train=True, download=True, transform=transform)
    
    # 随机取样
    indices = torch.randperm(len(dataset))[:Config.num_samples]
    dataloader = DataLoader(Subset(dataset, indices), batch_size=Config.batch_size)
    
    # 3. 扫描循环
    # 从 0 (数据) 扫描到 1000 (噪声)
    time_points = list(range(0, 1000, Config.scan_interval))
    stiffness_curve = []
    
    print(">>> Starting Scan Loop...")
    for t_val in tqdm(time_points, desc="Scanning t"):
        batch_stiff = []
        
        for x_0, _ in dataloader:
            x_0 = x_0.to(Config.device)
            B = x_0.shape[0]
            
            # 构造 t 时刻的加噪图 x_t
            t_batch = torch.full((B,), t_val, device=Config.device).long()
            noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, noise, t_batch)
            
            # 计算这一点的刚性
            norm = compute_jacobian_norm(unet, x_t, t_batch)
            batch_stiff.append(norm)
            
        # 记录平均刚性
        avg_stiff = np.mean(batch_stiff)
        stiffness_curve.append(avg_stiff)

    # 4. 绘图与保存
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, stiffness_curve, 'r-', linewidth=2, label='Jacobian Norm')
    plt.fill_between(time_points, stiffness_curve, color='red', alpha=0.1)
    
    plt.title("Diffusion Stiffness Spectrum (Where is the Chaos?)")
    plt.xlabel("Diffusion Time t (0=Data -> 1000=Noise)")
    plt.ylabel("|| d(eps)/dx || (Stiffness)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 寻找峰值
    max_stiff = max(stiffness_curve)
    max_t = time_points[stiffness_curve.index(max_stiff)]
    
    # 标注峰值
    plt.annotate(f'Max Stiffness: {max_stiff:.2f} @ t={max_t}', 
                 xy=(max_t, max_stiff), 
                 xytext=(max_t+50, max_stiff),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    save_path = os.path.join(Config.save_dir, "stiffness_spectrum.png")
    plt.savefig(save_path)
    
    print(f"\n📊 Analysis Complete.")
    print(f"   Max Stiffness: {max_stiff:.4f} at t = {max_t}")
    print(f"   ✅ Image saved to: {save_path}")
    
    # 给出建议
    print("\n💡 Recommendation:")
    print(f"   Your J-Net should focus on the range around t={max_t}.")
    print("   This is where the Euler method is most likely to fail due to high curvature.")

if __name__ == "__main__":
    run_stiffness_scan()