import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import matplotlib as mpl

# 绘图风格设置
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

# ==========================================
# 0. 配置 (Config) - 极简模式
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 物理参数 ===
    n_steps = 100        # 总步数
    batch_size = 1024    # 大批量，保证覆盖流形
    
    # === 训练参数 ===
    teacher_epochs = 2000 
    lr_teacher = 1e-3
    
    # === J-Net 核心设置 ===
    # 挑战模式：让 Teacher 走 40 步弯路，Student 走 1 步直路
    distill_skip = 40    
    student_epochs = 1500
    lr_student = 1e-3
    
    # === 物理约束 ===
    j_scale = 1.0          # 全功率
    energy_threshold = 1.0 # 允许大修 (因为我们要看明显的切弯效果)
    
    save_dir = "./toy_swiss_roll_final"

os.makedirs(Config.save_dir, exist_ok=True)
print(f"🚀 Toy Experiment initialized on: {Config.device}")

# ==========================================
# 1. 强力 MLP 架构 (Gaussian Fourier)
# ==========================================
class GaussianFourierProjection(nn.Module):
    """
    将低维坐标映射到高维，让 MLP 能学高频细节。
    """
    def __init__(self, embed_dim, scale=30.0, input_dim=2):
        super().__init__()
        # W: [embed_dim // 2, input_dim]
        # 修复：明确指定 input_dim，防止 t(1D) 和 x(2D) 冲突
        self.W = nn.Parameter(torch.randn(embed_dim // 2, input_dim) * scale, requires_grad=False)

    def forward(self, x):
        # x: [B, input_dim] -> proj: [B, embed_dim // 2]
        x_proj = x @ self.W.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimeMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 坐标嵌入: 2D -> 128
        self.coord_embed = GaussianFourierProjection(hidden_dim, scale=10.0, input_dim=2)
        
        # 时间嵌入: 1D -> 128
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(hidden_dim, scale=1.0, input_dim=1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 主干 ResNet MLP
        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        
        self.final = nn.Linear(hidden_dim, 2) # 输出 2D 速度向量

    def forward(self, x, t):
        # t: [B] -> [B, 1]
        t = t.view(-1, 1).float()
        
        h_x = self.coord_embed(x)
        h_t = self.time_mlp(t)
        
        h = h_x + h_t
        h = self.mid_layers(h) + h # Residual connection
        return self.final(h)

# ==========================================
# 2. 物理动力学
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
    核心：2D 空间的正交投影 + 能量门控
    """
    # 1. 计算 Teacher 方向 (归一化)
    v_norm = torch.norm(v_teacher, dim=1, keepdim=True) + 1e-6
    n = v_teacher / v_norm
    
    # 2. 正交投影: v_corr 必须垂直于 v_teacher
    # proj = (u . n) * n
    proj = (u_student * n).sum(dim=1, keepdim=True) * n
    v_corr = u_student - proj
    
    # 3. 能量门控: 防止修正过大飞出流形
    corr_norm = torch.norm(v_corr, dim=1, keepdim=True) + 1e-6
    threshold = Config.energy_threshold * v_norm
    gating = torch.clamp(threshold / corr_norm, max=1.0)
    
    # 4. 组合
    v_final = v_teacher + v_corr * gating * Config.j_scale
    return v_final

def get_swiss_roll(batch_size):
    x, _ = make_swiss_roll(n_samples=batch_size, noise=0.1) # 噪声小一点，流形更清晰
    x = x[:, [0, 2]] / 10.0 # 标准化到 [-1, 1] 附近
    return torch.from_numpy(x).float().to(Config.device)

# ==========================================
# 3. 训练循环 (Teacher & Student)
# ==========================================
def train_toy():
    # --- 1. Train Teacher ---
    print(">>> Training Teacher (Learning the Curve)...")
    teacher = TimeMLP().to(Config.device)
    opt = optim.Adam(teacher.parameters(), lr=Config.lr_teacher)
    
    for epoch in tqdm(range(Config.teacher_epochs), desc="Teacher"):
        x0 = get_swiss_roll(Config.batch_size)
        t = torch.randint(0, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, noise = dm.q_sample(x0, t)
        
        # Teacher 预测噪声 (Score Matching)
        pred_noise = teacher(xt, t)
        loss = F.mse_loss(pred_noise, noise)
        
        opt.zero_grad(); loss.backward(); opt.step()
    
    teacher.eval()
    
    # --- 2. Train Student ---
    print("\n>>> Training Student (Learning the Shortcut)...")
    student = TimeMLP().to(Config.device) # J-Net 结构同 Teacher
    opt_s = optim.Adam(student.parameters(), lr=Config.lr_student)
    
    for epoch in tqdm(range(Config.student_epochs), desc="Student"):
        x0 = get_swiss_roll(Config.batch_size)
        # 随机起点 (保证有足够空间走 skip 步)
        t_start = torch.randint(Config.distill_skip, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, _ = dm.q_sample(x0, t_start)
        
        # A. Teacher Ground Truth (走弯路)
        with torch.no_grad():
            x_target = xt.clone()
            for i in range(Config.distill_skip):
                t_curr = t_start - i
                pred = teacher(x_target, t_curr)
                # Euler Update
                x_target = x_target - pred * (1.0/Config.n_steps)
        
        # B. Student Prediction (走直路)
        v_teacher = teacher(xt, t_start)
        u_student = student(xt, t_start)
        
        # 应用正交修正
        v_corrected = compute_corrected_velocity(v_teacher, u_student)
        
        # 一步跨越
        dt_big = (1.0/Config.n_steps) * Config.distill_skip
        x_pred = xt - v_corrected * dt_big
        
        # Loss
        loss = F.mse_loss(x_pred, x_target)
        
        opt_s.zero_grad(); loss.backward(); opt_s.step()
        
    return teacher, student

# ==========================================
# 4. 可视化 A: 几何轨迹与流场
# ==========================================
def visualize_results(teacher, student):
    print("\n>>> Generating Geometric Visualization...")
    
    # 1. 准备网格 (用于画流场)
    grid_size = 100
    range_lim = 1.5
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_tensor = torch.tensor(np.stack([xx, yy], axis=-1)).float().view(-1, 2).to(Config.device)
    
    # 固定时间 t=50 (中间时刻，曲率最大)
    t = torch.full((grid_tensor.shape[0],), 50, device=Config.device).long()
    
    with torch.no_grad():
        v_t = teacher(grid_tensor, t)
        u_s = student(grid_tensor, t)
        v_final = compute_corrected_velocity(v_t, u_s)
        
    v_t_np = v_t.cpu().numpy().reshape(grid_size, grid_size, 2)
    v_f_np = v_final.cpu().numpy().reshape(grid_size, grid_size, 2)
    
    # 2. 准备轨迹测试点 (瑞士卷上的点)
    # 取一圈螺旋
    thetas = np.linspace(1.5 * np.pi, 3.5 * np.pi, 20)
    r = thetas * 0.5 
    circle_x = (r * np.cos(thetas) - r.mean()) / 3.0
    circle_y = (r * np.sin(thetas) - r.mean()) / 3.0
    
    start_points = torch.tensor(np.stack([circle_x, circle_y], axis=1)).float().to(Config.device)
    t_start_val = 50
    
    # Teacher 轨迹
    traj_t = [start_points.cpu().numpy()]
    curr = start_points.clone()
    for i in range(Config.distill_skip):
        with torch.no_grad(): pred = teacher(curr, torch.tensor([t_start_val - i]).to(Config.device))
        curr = curr - pred * (1.0/Config.n_steps)
        traj_t.append(curr.cpu().numpy())
    traj_t = np.array(traj_t) # [Steps, B, 2]
    
    # Student 轨迹 (一步)
    with torch.no_grad():
        vt = teacher(start_points, torch.tensor([t_start_val]).to(Config.device))
        us = student(start_points, torch.tensor([t_start_val]).to(Config.device))
        vc = compute_corrected_velocity(vt, us)
        # 一步到位
        end_s = start_points - vc * (1.0/Config.n_steps * Config.distill_skip)
    end_s = end_s.cpu().numpy()
    
    # Baseline (Euler) 轨迹 (一步)
    with torch.no_grad():
        vb = teacher(start_points, torch.tensor([t_start_val]).to(Config.device))
        end_b = start_points - vb * (1.0/Config.n_steps * Config.distill_skip)
    end_b = end_b.cpu().numpy()
    
    # 3. 绘图
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    
    # 背景数据分布
    data_bg = get_swiss_roll(5000).cpu().numpy()
    
    for i in range(3):
        ax[i].hist2d(data_bg[:,0], data_bg[:,1], bins=100, cmap='Greys', alpha=0.1)
        ax[i].set_xlim(-1.5, 1.5)
        ax[i].set_ylim(-1.5, 1.5)
    
    # Plot 1: Teacher Flow
    ax[0].streamplot(x, y, -v_t_np[:,:,0], -v_t_np[:,:,1], color='blue', density=1.0)
    ax[0].set_title("Teacher Flow (Original)", fontsize=14)
    
    # Plot 2: Student Flow
    ax[1].streamplot(x, y, -v_f_np[:,:,0], -v_f_np[:,:,1], color='red', density=1.0)
    ax[1].set_title("J-Net Corrected Flow (Straightened)", fontsize=14)
    
    # Plot 3: Trajectory Comparison
    for k in range(len(start_points)):
        # 蓝线：Teacher 弯路
        ax[2].plot(traj_t[:, k, 0], traj_t[:, k, 1], 'b-', linewidth=2, alpha=0.4)
        ax[2].scatter(traj_t[-1, k, 0], traj_t[-1, k, 1], c='blue', s=80, label='Teacher' if k==0 else "")
        
        # 绿叉：Baseline (纯直线，飞出去)
        ax[2].plot([start_points[k, 0].item(), end_b[k, 0].item()], 
                   [start_points[k, 1].item(), end_b[k, 1].item()], 'g--', alpha=0.5)
        ax[2].scatter(end_b[k, 0].item(), end_b[k, 1].item(), c='green', marker='x', s=80, label='Euler Baseline' if k==0 else "")

        # 红星：SDDPM (修正后的直线，拉回来)
        ax[2].plot([start_points[k, 0].item(), end_s[k, 0].item()], 
                   [start_points[k, 1].item(), end_s[k, 1].item()], 'r--', linewidth=2, alpha=0.8)
        ax[2].scatter(end_s[k, 0].item(), end_s[k, 1].item(), c='red', marker='*', s=200, label='SDDPM (Ours)' if k==0 else "")

    ax[2].set_title("Trajectory Correction (Skip=40)", fontsize=14)
    ax[2].legend()
    
    plt.savefig(f"{Config.save_dir}/toy_trajectory.png")
    print(f"✅ Visualization saved to {Config.save_dir}/toy_trajectory.png")

# ==========================================
# 5. 可视化 B: 动力学刚性分析 (Velocity Profile)
# ==========================================
def exp_velocity_profile_toy(teacher, student):
    print("\n>>> [Physics Check] Analyzing Kinetic Stiffness...")
    
    n_samples = 100
    x0 = get_swiss_roll(n_samples)
    t_max = Config.n_steps - 1
    x_start, _ = dm.q_sample(x0, torch.full((n_samples,), t_max, device=Config.device).long())
    
    steps_to_track = list(range(t_max, 0, -1))
    vel_norms_ddpm = []
    vel_norms_sddpm = []
    
    # --- DDPM ---
    curr_x = x_start.clone()
    for t_val in steps_to_track:
        t_batch = torch.full((n_samples,), t_val, device=Config.device).long()
        with torch.no_grad(): v = teacher(curr_x, t_batch)
        vel_norms_ddpm.append(torch.norm(v, dim=1).mean().item())
        curr_x = curr_x - v * (1.0 / Config.n_steps)

    # --- SDDPM ---
    curr_x = x_start.clone()
    for t_val in steps_to_track:
        t_batch = torch.full((n_samples,), t_val, device=Config.device).long()
        with torch.no_grad():
            v_t = teacher(curr_x, t_batch)
            u_s = student(curr_x, t_batch)
            v_final = compute_corrected_velocity(v_t, u_s)
        vel_norms_sddpm.append(torch.norm(v_final, dim=1).mean().item())
        curr_x = curr_x - v_final * (1.0 / Config.n_steps)
        
    # Stats
    var_d = np.var(vel_norms_ddpm)
    var_s = np.var(vel_norms_sddpm)
    
    # Plot
    plt.figure(figsize=(10, 6))
    x_axis = np.linspace(1, 0, len(steps_to_track))
    plt.plot(x_axis, vel_norms_ddpm, 'b-', label=f'DDPM (Var={var_d:.4f})', alpha=0.6)
    plt.plot(x_axis, vel_norms_sddpm, 'r-', label=f'SDDPM (Var={var_s:.4f})', linewidth=2)
    
    plt.title("Kinetic Stiffness Analysis (Velocity Magnitude Stability)")
    plt.xlabel("Diffusion Time")
    plt.ylabel("Velocity Norm ||v||")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.gca().invert_xaxis()
    
    plt.savefig(f"{Config.save_dir}/toy_velocity_profile.png")
    print(f"✅ Velocity Profile saved to {Config.save_dir}/toy_velocity_profile.png")

if __name__ == "__main__":
    teacher, student = train_toy()
    visualize_results(teacher, student)
    exp_velocity_profile_toy(teacher, student)
    print("\n🎉 All Done!")