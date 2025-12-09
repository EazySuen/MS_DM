import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DModel 
import shutil

# ==========================================
# 0. 配置 (Config) - 必须与数据生成一致
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 必须与 prepare_data.py 完全一致 ===
    distill_skip = 500 
    n_steps = 1000
    cache_dir = "./cifar_cache_ode_inversion" 
    
    # 确保训练时的步长公式与生成数据时一致
    # 我们的代码里已经是: x_pred = x_t - pred * total_dt
    # total_dt = (1/1000) * 500 = 0.5
    # 这与 prepare_data 里的循环累加是数学等价的。
    
    # === 训练参数 ===
    lr_j = 2e-4
    epochs_j = 80
    
    # === 物理约束参数 ===
    j_scale = 0.01           # 基础力度
    energy_threshold = 0.10 # 500步是大动作，允许修正量达到原始能量的 50%
    lambda_div = 1e-4       # 散度惩罚
    lambda_stiff = 1e-3     # 刚性惩罚
    
    # === 输出路径 ===
    base_dir = "./experiment_offline_500"
    dir_weights = f"{base_dir}/weights"
    dir_results = f"{base_dir}/results"
    path_student = f"{dir_weights}/student_500.pt"

# 初始化目录
print(f"🚀 Offline Training (Skip={Config.distill_skip}) initialized.")
if os.path.exists(Config.base_dir): shutil.rmtree(Config.base_dir) #以此为准，清空旧的
os.makedirs(Config.dir_weights, exist_ok=True)
os.makedirs(Config.dir_results, exist_ok=True)

# ==========================================
# 1. 模型定义 (Teacher & Student)
# ==========================================
class GoogleTeacherWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        print(">>> Loading SOTA Teacher...")
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(Config.device)
        self.unet.eval()
        for p in self.unet.parameters(): p.requires_grad = False
        self._mid_h = None

    def get_latent_dim(self):
        def _hook(module, inp, out): self._mid_h = out
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(Config.device)
            t = torch.tensor([0]).long().to(Config.device)
            handle = self.unet.mid_block.register_forward_hook(_hook)
            self.unet(dummy, t)
            handle.remove()
        return self._mid_h.shape[1]

    def get_mid_h(self, x, t):
        """只跑 Encoder，获取潜变量 h"""
        def _hook(module, inp, out): self._mid_h = out.clone()
        handle = self.unet.mid_block.register_forward_hook(_hook)
        with torch.no_grad():
            self.unet(x, t)
        handle.remove()
        return self._mid_h

    def run_decoder(self, x, t, h_injected):
        """只跑 Decoder，注入修正后的 h"""
        def _hook_replace(module, inp, out): return h_injected
        handle = self.unet.mid_block.register_forward_hook(_hook_replace)
        # 允许梯度回传
        out = self.unet(x, t).sample
        handle.remove()
        return out

class LatentJNet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, in_c))
        # 轻量级 ResNet
        self.net = nn.Sequential(
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1)
        )
        # 极小初始化 (打破死锁)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, t):
        # t 归一化后映射
        t_vec = t.view(-1, 1).float() / 1000.0 
        t_feat = self.time_mlp(t_vec)[:, :, None, None]
        h_in = h + t_feat
        # Tanh 保护
        return torch.tanh(self.net(h_in))

# ==========================================
# 2. 物理与数学函数 (修正版)
# ==========================================
def compute_corrected_latent(h, u):
    """正交投影 + 能量门控"""
    B_size = h.shape[0]
    # === 修复：使用 reshape 防止 view 报错 ===
    h_flat = h.reshape(B_size, -1)
    u_flat = u.reshape(B_size, -1)
    
    # 1. 计算方向
    h_norm = torch.norm(h_flat, dim=1, keepdim=True) + 1e-6
    n_flat = h_flat / h_norm
    
    # 2. 正交投影
    proj = (u_flat * n_flat).sum(dim=1, keepdim=True) * n_flat
    v_corr_flat = u_flat - proj
    
    # 3. 能量门控 (Energy Gating)
    v_norm = torch.norm(v_corr_flat, dim=1, keepdim=True) + 1e-6
    threshold = Config.energy_threshold * h_norm
    gating = torch.clamp(threshold / v_norm, max=1.0)
    
    v_corr = (v_corr_flat * gating).reshape(h.shape)
    
    return h + v_corr, v_corr

def compute_physics_losses(student, h, t):
    """计算 Div 和 Stiff"""
    h_in = h.detach().clone().requires_grad_(True)
    u = student(h_in, t)
    _, v_corr = compute_corrected_latent(h_in, u)
    
    z = torch.randn_like(v_corr)
    g = torch.autograd.grad(v_corr, h_in, z, create_graph=True, retain_graph=True)[0]
    
    div = torch.mean(torch.sum(g * z, dim=[1,2,3])**2)
    stiff = torch.mean(torch.sum(g**2, dim=[1,2,3]))
    return div, stiff

def normalize_to_img(tensor):
    return (tensor.clamp(-1, 1) * 0.5 + 0.5)

# ==========================================
# 3. 数据集加载器
# ==========================================
class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pt')]
        print(f"📂 Found {len(self.files)} cached batches in {cache_dir}")
        if len(self.files) == 0:
            raise FileNotFoundError("No .pt files found! Please run prepare_data.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载整个 Batch
        return torch.load(self.files[idx])

# ==========================================
# 4. 训练主程序
# ==========================================
def train_offline():
    # 1. 准备模型
    teacher = GoogleTeacherWrapper()
    latent_dim = teacher.get_latent_dim()
    student = LatentJNet(in_c=latent_dim).to(Config.device)
    
    optimizer = optim.Adam(student.parameters(), lr=Config.lr_j)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs_j)
    
    # 2. 准备数据
    dataset = CachedDataset(Config.cache_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    # 辅助 Loss 函数
    criterion_raw = nn.MSELoss(reduction='none')
    
    print("\n>>> Start Offline Training (Task-Aware + Physics Monitoring)...")
    
    for epoch in range(Config.epochs_j):
        student.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        # 记录每轮平均值
        ep_loss_high = []
        ep_loss_low = []
        
        for batch_data in pbar:
            # 解包
            x_t = batch_data["x_t"].squeeze(0).to(Config.device)
            t = batch_data["t"].squeeze(0).to(Config.device)
            x_target = batch_data["x_target"].squeeze(0).to(Config.device)
            
            # --- Forward ---
            h_teacher = teacher.get_mid_h(x_t, t)
            u = student(h_teacher, t)
            h_student, _ = compute_corrected_latent(h_teacher, u)
            pred_noise = teacher.run_decoder(x_t, t, h_student)
            
            total_dt = (1.0 / Config.n_steps) * Config.distill_skip
            x_pred = x_t - pred_noise * total_dt
            
            # --- Advanced Loss Calculation ---
            loss_per_sample = criterion_raw(x_pred, x_target).mean(dim=[1, 2, 3])
            
            mask_high = (t > 500).float()
            mask_low = (t <= 500).float()
            
            l_high = (loss_per_sample * mask_high).sum() / (mask_high.sum() + 1e-6)
            l_low = (loss_per_sample * mask_low).sum() / (mask_low.sum() + 1e-6)
            
            if mask_high.sum() > 0: ep_loss_high.append(l_high.item())
            if mask_low.sum() > 0: ep_loss_low.append(l_low.item())
            
            # 1:2 加权策略
            weighted_dist = (l_high + 2.0 * l_low) / 2.0 
            loss_dist = weighted_dist
            
            # Physics Loss
            div, stiff = compute_physics_losses(student, h_teacher.detach(), t)
            loss = loss_dist + Config.lambda_div * div + Config.lambda_stiff * stiff
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # === 实时显示所有关键指标 (包含 Div & Stiff) ===
            avg_h = np.mean(ep_loss_high[-10:]) if ep_loss_high else 0.0
            avg_l = np.mean(ep_loss_low[-10:]) if ep_loss_low else 0.0
            
            pbar.set_postfix({
                "H(1)": f"{avg_h:.3f}",   # Step 1 Loss
                "L(2)": f"{avg_l:.3f}",   # Step 2 Loss (重点关注)
                "Div": f"{div.item():.2e}", # 散度 (应该很小)
                "Stiff": f"{stiff.item():.2e}" # 刚性 (应该很小)
            })
            
        scheduler.step()
        torch.save(student.state_dict(), Config.path_student)
        
        # Viz
        # ... (scheduler.step() 和 torch.save() 保持不变) ...
        
        # === 核心修改：每 1 个 Epoch 详细可视化 ===
        # 频率：每个 Epoch 都保存
        if True: 
            with torch.no_grad():
                n_vis = 8 # 只看前8张
                
                # --- Row 1: 输入噪声点 (Input x_t) ---
                # 这是模型看到的起点。如果是 t=999，这里应该是纯噪声。
                row_input = normalize_to_img(x_t[:n_vis])
                
                # --- Row 2: Teacher 的 500 步真值 (Teacher Target) ---
                # 这是 Student 拼命要模仿的对象。
                # 如果这一行是清晰的，说明 Teacher 没问题；如果是糊的，说明 500 步本身就有截断误差。
                row_teacher = normalize_to_img(x_target[:n_vis])
                
                # --- Row 3: J-Net 的 1 步预测 (Student Prediction) ---
                # 这是你最关心的。如果这行全是噪点，说明模型炸了；如果是清晰的，说明成功了。
                row_student = normalize_to_img(x_pred[:n_vis])
                
                # --- Row 4: 差值热力图 (Difference) ---
                # |Student - Teacher|
                # 我们做一个自适应归一化：让当前Batch最大误差显示为白色，0显示为黑色
                diff = (x_pred[:n_vis] - x_target[:n_vis]).abs()
                row_diff = diff / (diff.max() + 1e-6)
                
                # --- 拼图 ---
                # 最终图片高度 = 4 * 32 = 128 像素
                grid = torch.cat([row_input, row_teacher, row_student, row_diff], dim=0)
                
                save_image(grid, f"{Config.dir_results}/epoch_{epoch+1}.png", nrow=n_vis, padding=2)
                
                print(f"📸 Visualization saved: epoch_{epoch+1}.png")
                
    print("✅ Offline Training Completed!")

if __name__ == "__main__":
    train_offline()