import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 0. 配置
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 512           # 模拟真实潜空间维度
    rank_true = 4       # 真实的刚性方向数量
    rank_model = 8      # LoRA 的秩 (稍微大一点点)
    
    noise_level = 20.0  # 梯度噪声强度
    lr = 0.05
    steps = 1000
    
    stiffness = 10.0

# ==========================================
# 1. 构造高刚性环境
# ==========================================
def get_stiff_matrix():
    # 构造一个对角矩阵，只有前4个值很大(刚性)，其余很小
    eigs = torch.cat([
        torch.ones(Config.rank_true) * Config.stiffness,
        torch.randn(Config.dim - Config.rank_true) * 0.1 
    ]).to(Config.device)
    
    # 随机旋转
    Q = torch.linalg.qr(torch.randn(Config.dim, Config.dim, device=Config.device))[0]
    A = Q @ torch.diag(eigs) @ Q.T
    return A, eigs 

# ==========================================
# 2. 模型定义
# ==========================================
class FullRankJ(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
    
    def get_matrix(self):
        return self.W - self.W.T

class SymplecticLoRA(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)
        
    def get_matrix(self):
        return self.U @ self.V.T - self.V @ self.U.T

# ==========================================
# 3. 训练与分析
# ==========================================
def analyze_spectral_dynamics():
    print(f"🚀 Starting Spectral Analysis (Dim={Config.dim}, Rank={Config.rank_true})...")
    
    A_target, true_eigs = get_stiff_matrix()
    
    model_full = FullRankJ(Config.dim).to(Config.device)
    model_lora = SymplecticLoRA(Config.dim, Config.rank_model).to(Config.device)
    
    opt_full = optim.SGD(model_full.parameters(), lr=Config.lr)
    opt_lora = optim.SGD(model_lora.parameters(), lr=Config.lr)
    
    loss_history_full = []
    loss_history_lora = []
    
    # 进度条
    pbar = tqdm(range(Config.steps))
    
    for i in pbar:
        # --- Train Full ---
        opt_full.zero_grad()
        J_full = model_full.get_matrix()
        loss_f = torch.norm(A_target + J_full)**2 
        loss_f.backward()
        
        # 注入噪声 + 梯度裁剪
        for param in model_full.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * Config.noise_level
                torch.nn.utils.clip_grad_norm_([param], 1.0) 
        
        opt_full.step()
        loss_history_full.append(loss_f.item())
        
        # --- Train LoRA ---
        opt_lora.zero_grad()
        J_lora = model_lora.get_matrix()
        loss_l = torch.norm(A_target + J_lora)**2
        loss_l.backward()
        
        for param in model_lora.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * Config.noise_level
                torch.nn.utils.clip_grad_norm_([param], 1.0)
            
        opt_lora.step()
        loss_history_lora.append(loss_l.item())
        
        # === 实时监控 ===
        if i % 10 == 0:
            if torch.isnan(loss_f) or torch.isnan(loss_l):
                print(f"\n💀 Numerical Explosion detected at step {i}!")
                break
            norm_full = torch.norm(J_full).item()
            norm_lora = torch.norm(J_lora).item()
            pbar.set_postfix({
                "L_Full": f"{loss_f.item():.2f}", 
                "L_LoRA": f"{loss_l.item():.2f}",
                "Norm_F": f"{norm_full:.1f}",
                "Norm_L": f"{norm_lora:.1f}"
            })
            
    print("\n>>> Calculation Finished. Computing Eigenvalues (on CPU)...")

    # 3. 谱分析
    with torch.no_grad():
        A_cpu = A_target.cpu()
        M_final_full = (A_target + model_full.get_matrix()).cpu()
        M_final_lora = (A_target + model_lora.get_matrix()).cpu()
        
        if torch.isnan(M_final_full).any():
            M_final_full = torch.zeros_like(M_final_full)
            
        eig_orig = torch.linalg.eigvals(A_cpu)
        eig_full = torch.linalg.eigvals(M_final_full)
        eig_lora = torch.linalg.eigvals(M_final_lora)

    print(">>> Plotting...")

    # 4. 绘图
    plt.figure(figsize=(18, 6))
    
    # Plot A: Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss_history_full, label='Full Rank', alpha=0.6, color='gray')
    plt.plot(loss_history_lora, label='Symplectic LoRA', linewidth=2, color='red')
    plt.title("Convergence Speed under High Noise")
    plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    # Plot B: Eigenvalues
    plt.subplot(1, 3, 2)
    plt.scatter(eig_orig.real, eig_orig.imag, c='black', label='Original (Stiff)', marker='x', s=50, alpha=0.5)
    plt.scatter(eig_full.real, eig_full.imag, c='blue', label='Full Rank', s=15, alpha=0.3)
    plt.scatter(eig_lora.real, eig_lora.imag, c='red', label='LoRA', s=30, alpha=0.8)
    plt.title("Eigenvalue Spectrum")
    plt.xlabel("Real"); plt.ylabel("Imag")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    # Plot C: Singular Values
    plt.subplot(1, 3, 3)
    _, S_full, _ = torch.linalg.svd(model_full.get_matrix().cpu()) 
    _, S_lora, _ = torch.linalg.svd(model_lora.get_matrix().cpu())
    
    # === 修复点：加上 .detach() ===
    plt.plot(S_full.detach().numpy(), label='Full Rank SV', color='blue')
    plt.plot(S_lora.detach().numpy(), label='LoRA SV', color='red', linestyle='--')
    
    plt.title("Learned Singular Values (Rank)")
    plt.xlabel("Index"); plt.ylabel("Magnitude")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spectral_analysis_fixed.png")
    print("✅ Done. Check 'spectral_analysis_fixed.png'")

if __name__ == "__main__":
    analyze_spectral_dynamics()