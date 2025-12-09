import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# ==========================================
# 0. 配置 (Config)
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 512 # 2D 数据可以大 Batch
    
    # Teacher 训练
    teacher_epochs = 2000 # MLP 跑得快，多训点保证完美
    lr_teacher = 1e-3
    
    # Student 训练
    distill_skip = 20   # 100步里跨20步 (相当于 CIFAR 的 200/1000)
    student_epochs = 1000
    lr_student = 1e-3
    
    # 物理参数
    j_scale = 1.0       # 2D 空间比较简单，全功率
    energy_threshold = 0.5 # 允许大修
    
    save_dir = "./toy_swiss_roll"

os.makedirs(Config.save_dir, exist_ok=True)

# ==========================================
# 1. 强力 MLP 架构 (修复版)
# ==========================================
class GaussianFourierProjection(nn.Module):
    """
    将低维坐标映射到高维，让 MLP 能学高频细节。
    已修复：支持自定义 input_dim (1 或 2)
    """
    def __init__(self, embed_dim, scale=30.0, input_dim=2):
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
    """强力 Teacher/Student"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 坐标嵌入: 输入是 2D (x, y), 输出 128
        self.coord_embed = GaussianFourierProjection(hidden_dim, scale=10.0, input_dim=2)
        
        # 时间嵌入: 输入是 1D (t), 输出 128
        # === 关键修复：指定 input_dim=1 ===
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(hidden_dim, scale=1.0, input_dim=1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 主干网络 (ResNet MLP)
        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        
        self.final = nn.Linear(hidden_dim, 2) # 输出 2D 向量

    def forward(self, x, t):
        # x: [B, 2]
        # t: [B] -> [B, 1]
        t = t.view(-1, 1).float()
        
        h_x = self.coord_embed(x) # [B, 128]
        h_t = self.time_mlp(t)    # [B, 128]
        
        h = h_x + h_t
        h = self.mid_layers(h) + h # Residual
        return self.final(h)

# ==========================================
# 2. 物理动力学 (2D 专用)
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
    """
    2D 空间的正交投影修正
    v_teacher: [B, 2]
    u_student: [B, 2]
    """
    # 1. 计算 Teacher 方向
    v_norm = torch.norm(v_teacher, dim=1, keepdim=True) + 1e-6
    n = v_teacher / v_norm
    
    # 2. 正交投影
    proj = (u_student * n).sum(dim=1, keepdim=True) * n
    v_corr = u_student - proj
    
    # 3. 能量门控
    corr_norm = torch.norm(v_corr, dim=1, keepdim=True) + 1e-6
    threshold = Config.energy_threshold * v_norm
    gating = torch.clamp(threshold / corr_norm, max=1.0)
    
    v_final = v_teacher + v_corr * gating * Config.j_scale
    return v_final

# ==========================================
# 3. 训练与测试
# ==========================================
def get_swiss_roll(batch_size):
    x, _ = make_swiss_roll(n_samples=batch_size, noise=0.5)
    x = x[:, [0, 2]] / 10.0 # 只取 x, z 坐标并缩放
    return torch.from_numpy(x).float().to(Config.device)

def train_toy():
    # 1. Train Teacher
    print(">>> Training Strong Teacher...")
    teacher = TimeMLP().to(Config.device)
    opt = optim.Adam(teacher.parameters(), lr=Config.lr_teacher)
    
    for epoch in tqdm(range(Config.teacher_epochs)):
        x0 = get_swiss_roll(Config.batch_size)
        t = torch.randint(0, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, noise = dm.q_sample(x0, t)
        
        pred_noise = teacher(xt, t)
        loss = F.mse_loss(pred_noise, noise)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    teacher.eval()
    
    # 2. Train Student (Offline Distillation)
    print("\n>>> Training Student (J-Net)...")
    student = TimeMLP().to(Config.device) # J-Net 结构同 Teacher
    opt_s = optim.Adam(student.parameters(), lr=Config.lr_student)
    
    # 离线数据太快了，直接在线生成吧，没必要存硬盘
    for epoch in tqdm(range(Config.student_epochs)):
        # 生成数据对
        x0 = get_swiss_roll(Config.batch_size)
        # 随机选起点 t (比如 50 -> 30)
        t_start = torch.randint(Config.distill_skip, Config.n_steps, (Config.batch_size,), device=Config.device)
        
        # 构造 x_t
        xt, _ = dm.q_sample(x0, t_start)
        
        # Teacher 走 skip 步 (GT)
        with torch.no_grad():
            x_target = xt.clone()
            for i in range(Config.distill_skip):
                t_curr = t_start - i
                pred = teacher(x_target, t_curr)
                # Euler: x_{t-1} = x_t - eps * dt (简化公式)
                # 实际上 DDPM 是: x_{t-1} = (x_t - beta * eps / sqrt(1-alpha)) / sqrt(alpha)
                # 这里为了简化物理，我们假设 Teacher 预测的是速度 v
                # 如果要严谨，应该用 DDIM 公式。
                # 这里用最简单的 ODE 近似: x - pred * (1/N)
                x_target = x_target - pred * (1.0/Config.n_steps)
        
        # Student 修正
        v_teacher = teacher(xt, t_start)
        u_student = student(xt, t_start)
        v_corrected = compute_corrected_velocity(v_teacher, u_student)
        
        # 大步更新
        dt_big = (1.0/Config.n_steps) * Config.distill_skip
        x_pred = xt - v_corrected * dt_big
        
        loss = F.mse_loss(x_pred, x_target)
        
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
        
    return teacher, student

# ==========================================
# 4. 核心可视化 (2D 矢量场)
# ==========================================
def visualize_results(teacher, student):
    print(">>> Visualizing Vector Fields...")
    
    # 造网格
    grid_size = 100
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_tensor = torch.tensor(np.stack([xx, yy], axis=-1)).float().view(-1, 2).to(Config.device)
    
    # 选一个时间点 t=50 (中间时刻)
    t = torch.full((grid_tensor.shape[0],), 50, device=Config.device).long()
    
    with torch.no_grad():
        v_t = teacher(grid_tensor, t)
        u_s = student(grid_tensor, t)
        v_s = compute_corrected_velocity(v_t, u_s)
        
    v_t = v_t.cpu().numpy().reshape(grid_size, grid_size, 2)
    v_s = v_s.cpu().numpy().reshape(grid_size, grid_size, 2)
    
    # 画图
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Teacher Field
    strm1 = ax[0].streamplot(x, y, -v_t[:,:,0], -v_t[:,:,1], color='blue', density=1.5)
    ax[0].set_title("Teacher Vector Field (Curved)")
    
    # 2. Student Field
    strm2 = ax[1].streamplot(x, y, -v_s[:,:,0], -v_s[:,:,1], color='red', density=1.5)
    ax[1].set_title("SDDPM Corrected Field (Should be Straighter)")
    
    # 3. 采样结果对比 (起点相同，走一步)
    # 取一圈圆形的噪声
    thetas = np.linspace(0, 2*np.pi, 20)
    circle_x = 1.0 * np.cos(thetas)
    circle_y = 1.0 * np.sin(thetas)
    start_points = torch.tensor(np.stack([circle_x, circle_y], axis=1)).float().to(Config.device)
    
    # Teacher 走 20 小步
    traj_t = [start_points.cpu().numpy()]
    curr = start_points.clone()
    t_val = 50
    for _ in range(Config.distill_skip):
        with torch.no_grad(): pred = teacher(curr, torch.tensor([t_val]).to(Config.device))
        curr = curr - pred * (1.0/Config.n_steps)
        t_val -= 1
        traj_t.append(curr.cpu().numpy())
        
    # SDDPM 走 1 大步
    with torch.no_grad():
        vt = teacher(start_points, torch.tensor([50]).to(Config.device))
        us = student(start_points, torch.tensor([50]).to(Config.device))
        vc = compute_corrected_velocity(vt, us)
        end_s = start_points - vc * (1.0/Config.n_steps * Config.distill_skip)
    
    # 绘制轨迹
    traj_t = np.array(traj_t) # [Steps, B, 2]
    for i in range(len(start_points)):
        # 蓝线：Teacher 细碎的弯路
        ax[2].plot(traj_t[:, i, 0], traj_t[:, i, 1], 'b-', alpha=0.5, linewidth=1)
        # 蓝点：Teacher 终点
        ax[2].scatter(traj_t[-1, i, 0], traj_t[-1, i, 1], c='blue', s=30)
        
        # 红虚线：SDDPM 直线一步
        ax[2].plot([start_points[i, 0].item(), end_s[i, 0].item()], 
                   [start_points[i, 1].item(), end_s[i, 1].item()], 'r--', alpha=0.8)
        # 红星：SDDPM 终点
        ax[2].scatter(end_s[i, 0].item(), end_s[i, 1].item(), c='red', marker='*', s=100)

    ax[2].set_title("Trajectory Comparison (Blue=Teacher, Red=Ours)")
    
    plt.savefig(f"{Config.save_dir}/toy_result.png")
    print(f"✅ Visualization saved to {Config.save_dir}/toy_result.png")

if __name__ == "__main__":
    teacher, student = train_toy()
    visualize_results(teacher, student)