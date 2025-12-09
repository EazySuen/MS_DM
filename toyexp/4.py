import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import matplotlib as mpl

# 绘图风格
mpl.rcParams['figure.dpi'] = 150
# 开启 3D 绘图支持
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. 配置 (Config)
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 512
    
    # === 关键：维度提升 ===
    data_dim = 64      # 我们在 64 维空间做实验
    manifold_dim = 2   # 数据的本质维度是 2 (瑞士卷)
    
    # 训练参数
    teacher_epochs = 2000 
    lr_teacher = 1e-3
    
    distill_skip = 20  
    student_epochs = 1500 
    lr_student = 1e-3
    
    # 物理参数
    j_scale = 1.0       
    energy_threshold = 0.5 
    
    save_dir = "./toy_64d_swiss_roll"

os.makedirs(Config.save_dir, exist_ok=True)
print(f"🚀 High-Dim Toy Experiment (64D) initialized on: {Config.device}")

# ==========================================
# 1. 数据生成器 (高维嵌入)
# ==========================================
class HighDimSwissRoll:
    def __init__(self):
        # 生成一个随机的正交投影矩阵 (2 -> 64)
        # 保证数据在高维空间保持几何结构
        self.projection = torch.randn(Config.manifold_dim, Config.data_dim).to(Config.device)
        self.projection = F.normalize(self.projection, dim=1) # 归一化

    def get_batch(self, batch_size):
        # 1. 生成原始 2D 瑞士卷 (x, z)
        data, color = make_swiss_roll(n_samples=batch_size, noise=0.1)
        data = data[:, [0, 2]] / 10.0 # [B, 2]
        data_tensor = torch.from_numpy(data).float().to(Config.device)
        
        # 2. 投影到 64 维
        # x_64 = x_2 @ Proj
        high_dim_data = data_tensor @ self.projection
        
        return high_dim_data, color # 返回 color 用于可视化时的颜色映射

dataset_gen = HighDimSwissRoll()

# ==========================================
# 2. 模型定义 (适配 64D)
# ==========================================
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0, input_dim=64):
        super().__init__()
        # 随机固定权重，不可训练
        # W shape: [embed_dim // 2, input_dim]
        self.W = nn.Parameter(torch.randn(embed_dim // 2, input_dim) * scale, requires_grad=False)

    def forward(self, x):
        # x: [B, input_dim]
        # x_proj: [B, embed_dim // 2]
        x_proj = x @ self.W.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimeMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256): # 加宽网络适应高维
        super().__init__()
        # 坐标嵌入: 输入维度动态
        self.coord_embed = GaussianFourierProjection(hidden_dim, scale=5.0, input_dim=input_dim)
        
        # 时间嵌入: 输入维度固定为 1
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(hidden_dim, scale=1.0, input_dim=1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        
        self.final = nn.Linear(hidden_dim, input_dim) # 输出高维向量

    def forward(self, x, t):
        t = t.view(-1, 1).float()
        h = self.coord_embed(x) + self.time_mlp(t)
        h = self.mid_layers(h) + h
        return self.final(h)

# ==========================================
# 3. 物理动力学 & 训练
# ==========================================
class DiffusionManager:
    def __init__(self):
        self.betas = torch.linspace(1e-4, 0.02, Config.n_steps).to(Config.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise

dm = DiffusionManager()

def compute_corrected_velocity(v_teacher, u_student):
    """高维正交投影修正"""
    # 1. 归一化方向
    v_norm = torch.norm(v_teacher, dim=1, keepdim=True) + 1e-6
    n = v_teacher / v_norm
    
    # 2. 投影 (在高维空间中，u在v方向的分量)
    proj = (u_student * n).sum(dim=1, keepdim=True) * n
    v_corr = u_student - proj
    
    # 3. 能量门控
    corr_norm = torch.norm(v_corr, dim=1, keepdim=True) + 1e-6
    threshold = Config.energy_threshold * v_norm
    gating = torch.clamp(threshold / corr_norm, max=1.0)
    
    return v_teacher + v_corr * gating * Config.j_scale

def train_toy():
    # 1. Teacher
    print(">>> Training Teacher (64D)...")
    teacher = TimeMLP(input_dim=Config.data_dim).to(Config.device)
    opt = optim.Adam(teacher.parameters(), lr=Config.lr_teacher)
    
    for epoch in tqdm(range(Config.teacher_epochs)):
        x0, _ = dataset_gen.get_batch(Config.batch_size)
        t = torch.randint(0, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, noise = dm.q_sample(x0, t)
        
        pred = teacher(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        
    teacher.eval()
    
    # 2. Student
    print("\n>>> Training Student (J-Net)...")
    student = TimeMLP(input_dim=Config.data_dim).to(Config.device)
    opt_s = optim.Adam(student.parameters(), lr=Config.lr_student)
    
    for epoch in tqdm(range(Config.student_epochs)):
        x0, _ = dataset_gen.get_batch(Config.batch_size)
        t_start = torch.randint(Config.distill_skip, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, _ = dm.q_sample(x0, t_start)
        
        # GT
        with torch.no_grad():
            x_target = xt.clone()
            for i in range(Config.distill_skip):
                pred = teacher(x_target, t_start - i)
                x_target = x_target - pred * (1.0/Config.n_steps)
        
        # Student
        vt = teacher(xt, t_start)
        us = student(xt, t_start)
        vc = compute_corrected_velocity(vt, us)
        
        dt_big = (1.0/Config.n_steps) * Config.distill_skip
        x_pred = xt - vc * dt_big
        
        loss = F.mse_loss(x_pred, x_target)
        opt_s.zero_grad(); loss.backward(); opt_s.step()
        
    return teacher, student

# ==========================================
# 4. 3D PCA 可视化 (降维打击)
# ==========================================
def visualize_3d_pca(teacher, student):
    print("\n>>> Generating 3D PCA Visualization...")
    
    # 1. 准备测试数据 (取一部分点，按颜色区分流形位置)
    n_samples = 500
    x0_high, colors = dataset_gen.get_batch(n_samples)
    
    # 2. 构造起点 (加噪到 t=50)
    # 50 步是一个比较模糊的中间状态
    t_val = 50
    t_batch = torch.full((n_samples,), t_val, device=Config.device).long()
    xt_high, _ = dm.q_sample(x0_high, t_batch)
    
    # 3. 收集轨迹
    # Teacher 轨迹 (20步)
    traj_t = [xt_high.cpu().numpy()]
    curr = xt_high.clone()
    for i in range(Config.distill_skip):
        with torch.no_grad(): pred = teacher(curr, t_batch - i)
        curr = curr - pred * (1.0/Config.n_steps)
        traj_t.append(curr.cpu().numpy())
    
    # SDDPM 轨迹 (1步)
    with torch.no_grad():
        vt = teacher(xt_high, t_batch)
        us = student(xt_high, t_batch)
        vc = compute_corrected_velocity(vt, us)
        end_s = xt_high - vc * (1.0/Config.n_steps * Config.distill_skip)
    
    # 4. PCA 降维 (64D -> 3D)
    # 我们把所有点 (起点、过程、终点) 放在一起做 PCA，保证坐标系一致
    all_points = np.concatenate([np.array(traj_t).reshape(-1, Config.data_dim), end_s.cpu().numpy()], axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_points)
    
    # 转换轨迹
    traj_t_pca = np.array([pca.transform(step) for step in traj_t]) # [Steps, B, 3]
    start_pca = traj_t_pca[0]
    end_t_pca = traj_t_pca[-1]
    end_s_pca = pca.transform(end_s.cpu().numpy())
    
    # 5. 绘图 (3D Scatter)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画一部分样本的轨迹 (太乱了看不清，只画前 50 个)
    for i in range(50):
        # 蓝线: Teacher
        ax.plot(traj_t_pca[:, i, 0], traj_t_pca[:, i, 1], traj_t_pca[:, i, 2], 
                color='blue', alpha=0.3, linewidth=0.5)
        
        # 红虚线: SDDPM (起点 -> 终点)
        ax.plot([start_pca[i,0], end_s_pca[i,0]], 
                [start_pca[i,1], end_s_pca[i,1]], 
                [start_pca[i,2], end_s_pca[i,2]], 
                color='red', linestyle='--', alpha=0.6)
        
        # 终点
        ax.scatter(end_t_pca[i,0], end_t_pca[i,1], end_t_pca[i,2], c='blue', s=20, label='Teacher' if i==0 else "")
        ax.scatter(end_s_pca[i,0], end_s_pca[i,1], end_s_pca[i,2], c='red', marker='*', s=50, label='SDDPM' if i==0 else "")

    # 视角调整
    ax.view_init(elev=30, azim=45)
    ax.set_title("64D Trajectory Projected to 3D (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    # ax.legend() # 图例太多会乱
    
    plt.savefig(f"{Config.save_dir}/high_dim_pca.png")
    print(f"✅ Visualization saved to {Config.save_dir}/high_dim_pca.png")

if __name__ == "__main__":
    # === 修正函数名调用 ===
    teacher, student = train_toy()
    visualize_3d_pca(teacher, student)