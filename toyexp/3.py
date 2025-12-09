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

# 设置绘图风格
mpl.rcParams['figure.dpi'] = 150

# ==========================================
# 0. 配置 (Config)
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 512
    
    # Teacher 训练
    teacher_epochs = 2000 
    lr_teacher = 1e-3
    
    # Student 训练
    distill_skip = 20   
    student_epochs = 1000
    lr_student = 1e-3
    
    # 物理参数
    j_scale = 1.0       
    energy_threshold = 0.5 
    
    save_dir = "./toy_swiss_roll"

os.makedirs(Config.save_dir, exist_ok=True)
print(f"🚀 Toy Experiment initialized on: {Config.device}")

# ==========================================
# 1. 强力 MLP 架构 (已修复维度 Bug)
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
    2D 空间的正交投影修正
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

def get_swiss_roll(batch_size):
    x, _ = make_swiss_roll(n_samples=batch_size, noise=0.5)
    x = x[:, [0, 2]] / 10.0 # 只取 x, z 坐标并缩放
    return torch.from_numpy(x).float().to(Config.device)

# ==========================================
# 3. 训练函数
# ==========================================
def train_toy():
    # --- 1. Train Teacher ---
    print(">>> Training Strong Teacher...")
    teacher = TimeMLP().to(Config.device)
    opt = optim.Adam(teacher.parameters(), lr=Config.lr_teacher)
    
    for epoch in tqdm(range(Config.teacher_epochs), desc="Teacher"):
        x0 = get_swiss_roll(Config.batch_size)
        t = torch.randint(0, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, noise = dm.q_sample(x0, t)
        
        # DDPM Loss: 预测噪声
        # 注意：这里我们简化处理，假设模型直接预测 drift (速度场)
        # 为了方便可视化流场，我们直接回归 noise 其实等价于回归 score
        pred_noise = teacher(xt, t)
        loss = F.mse_loss(pred_noise, noise)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    teacher.eval()
    
    # --- 2. Train Student ---
    print("\n>>> Training Student (J-Net)...")
    student = TimeMLP().to(Config.device) # J-Net 结构同 Teacher
    opt_s = optim.Adam(student.parameters(), lr=Config.lr_student)
    
    for epoch in tqdm(range(Config.student_epochs), desc="Student"):
        # 生成数据
        x0 = get_swiss_roll(Config.batch_size)
        t_start = torch.randint(Config.distill_skip, Config.n_steps, (Config.batch_size,), device=Config.device)
        xt, _ = dm.q_sample(x0, t_start)
        
        # Teacher GT (走 skip 步)
        with torch.no_grad():
            x_target = xt.clone()
            for i in range(Config.distill_skip):
                t_curr = t_start - i
                pred = teacher(x_target, t_curr)
                # Euler Update
                x_target = x_target - pred * (1.0/Config.n_steps)
        
        # Student Prediction
        v_teacher = teacher(xt, t_start)
        u_student = student(xt, t_start)
        
        # 应用正交修正
        v_corrected = compute_corrected_velocity(v_teacher, u_student)
        
        # Student 走 1 大步
        dt_big = (1.0/Config.n_steps) * Config.distill_skip
        x_pred = xt - v_corrected * dt_big
        
        loss = F.mse_loss(x_pred, x_target)
        
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
        
    return teacher, student

# ==========================================
# 4. 高级可视化 (Advanced Visualization)
# ==========================================
def advanced_toy_visualization(teacher, student):
    print("\n>>> Running Advanced Toy Visualizations...")
    
    # 1. 准备网格
    grid_size = 100
    range_lim = 1.5
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_tensor = torch.tensor(np.stack([xx, yy], axis=-1)).float().view(-1, 2).to(Config.device)
    
    t = torch.full((grid_tensor.shape[0],), 50, device=Config.device).long()
    
    with torch.no_grad():
        v_t = teacher(grid_tensor, t)
        u_s = student(grid_tensor, t)
        v_final = compute_corrected_velocity(v_t, u_s)
        # 修正场向量
        v_c = v_final - v_t
        
    v_t_np = v_t.cpu().numpy().reshape(grid_size, grid_size, 2)
    v_c_np = v_c.cpu().numpy().reshape(grid_size, grid_size, 2)
    v_f_np = v_final.cpu().numpy().reshape(grid_size, grid_size, 2)
    
    # 2. 准备轨迹点
    thetas = np.linspace(0, 2*np.pi, 20)
    circle_x = 1.0 * np.cos(thetas)
    circle_y = 1.0 * np.sin(thetas)
    start_points = torch.tensor(np.stack([circle_x, circle_y], axis=1)).float().to(Config.device)
    
    # Teacher Trajectory
    traj_t = [start_points.cpu().numpy()]
    curr = start_points.clone()
    t_val = 50
    for _ in range(Config.distill_skip):
        with torch.no_grad(): pred = teacher(curr, torch.tensor([t_val]).to(Config.device))
        curr = curr - pred * (1.0/Config.n_steps)
        t_val -= 1
        traj_t.append(curr.cpu().numpy())
    
    # Student Step
    with torch.no_grad():
        vt = teacher(start_points, torch.tensor([50]).to(Config.device))
        us = student(start_points, torch.tensor([50]).to(Config.device))
        vc = compute_corrected_velocity(vt, us)
        end_s = start_points - vc * (1.0/Config.n_steps * Config.distill_skip)

    # 3. 绘图
    fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    
    # 真实数据分布背景
    data = get_swiss_roll(2000).cpu().numpy()
    
    # Plot A: Correction Field (J-Net 往哪推)
    ax[0].hist2d(data[:,0], data[:,1], bins=50, cmap='Greys', alpha=0.3)
    mag = np.sqrt(v_c_np[:,:,0]**2 + v_c_np[:,:,1]**2)
    # 只画修正力度够大的地方
    ax[0].streamplot(x, y, -v_c_np[:,:,0], -v_c_np[:,:,1], color=mag, cmap='autumn', density=1.5)
    ax[0].set_title("Correction Field (J-Net Pushes Particles Inward)")
    
    # Plot B: Flow Comparison (Blue=Teacher, Red=Student)
    ax[1].hist2d(data[:,0], data[:,1], bins=50, cmap='Greys', alpha=0.3)
    # Teacher (Blue)
    ax[1].streamplot(x, y, -v_t_np[:,:,0], -v_t_np[:,:,1], color='blue', density=0.8, linewidth=0.5, arrowsize=0.5)
    # Student (Red) - 应该更直
    ax[1].streamplot(x, y, -v_f_np[:,:,0], -v_f_np[:,:,1], color='red', density=0.8, linewidth=1.0)
    ax[1].set_title("Flow Comparison (Blue: Curved / Red: Straightened)")
    
    # Plot C: Trajectory Check
    traj_t = np.array(traj_t)
    for i in range(len(start_points)):
        # Teacher Path
        ax[2].plot(traj_t[:, i, 0], traj_t[:, i, 1], 'b-', alpha=0.4)
        ax[2].scatter(traj_t[-1, i, 0], traj_t[-1, i, 1], c='blue', s=30, label='Teacher End' if i==0 else "")
        # Student Step
        ax[2].plot([start_points[i, 0].item(), end_s[i, 0].item()], 
                   [start_points[i, 1].item(), end_s[i, 1].item()], 'r--', alpha=0.8)
        ax[2].scatter(end_s[i, 0].item(), end_s[i, 1].item(), c='red', marker='*', s=150, label='J-Net End' if i==0 else "")
    
    ax[2].set_title("Trajectory: J-Net (Red) hits the Blue Target in 1 step!")
    ax[2].legend()
    
    plt.savefig(f"{Config.save_dir}/toy_advanced_vis.png")
    print(f"✅ Saved to {Config.save_dir}/toy_advanced_vis.png")

if __name__ == "__main__":
    # 1. 训练
    teacher_model, student_model = train_toy()
    
    # 2. 可视化
    advanced_toy_visualization(teacher_model, student_model)