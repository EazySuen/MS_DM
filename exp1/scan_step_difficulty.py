import os
# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import math

# === 引入指标库 ===
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    print("✅ Metrics libraries loaded.")
except ImportError:
    print("❌ Error: torchmetrics not found. Please run: pip install torchmetrics torch-fidelity")
    exit()

# ==========================================
# 0. 配置 (Config)
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 32
    channels = 3
    batch_size = 100 
    
    # 权重路径 (指向你训练好的文件)
    base_dir = "./experiment_offline_500"
    path_student = f"{base_dir}/weights/student_500.pt"
    
    # 扫描的步数列表 (从 2 到 50，重点关注低步数)
    # 这里的步数 N 意味着：把 Teacher 的 1000 步压缩为 N 步走完
    # 修改这里：
    # 既然 50 步还很轻松，我们就测 100, 200, 400, 500
    scan_steps = [50, 100, 150, 200, 300, 400, 500]
    
    # 评测样本量
    num_samples = 2000
    
    # 物理参数
    n_steps = 1000
    j_scale = 1.0
    energy_threshold = 0.50
    
    save_dir = "./step_difficulty_scan"

os.makedirs(Config.save_dir, exist_ok=True)
print(f"🚀 Step Difficulty Scan initialized on: {Config.device}")

# ==========================================
# 1. 模型定义 (这 5 个是你之前缺的)
# ==========================================

class GoogleTeacherWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(Config.device)
        self.unet.eval()
        self._mid_h = None

    def get_latent_dim(self):
        def _hook(m, i, o): self._mid_h = o
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(Config.device)
            t = torch.tensor([0]).long().to(Config.device)
            handle = self.unet.mid_block.register_forward_hook(_hook)
            self.unet(dummy, t)
            handle.remove()
        return self._mid_h.shape[1]

    def get_mid_h(self, x, t):
        def _hook(m, i, o): self._mid_h = o.clone()
        handle = self.unet.mid_block.register_forward_hook(_hook)
        with torch.no_grad(): self.unet(x, t)
        handle.remove()
        return self._mid_h

    def run_decoder(self, x, t, h_injected):
        def _hook(m, i, o): return h_injected
        handle = self.unet.mid_block.register_forward_hook(_hook)
        out = self.unet(x, t).sample
        handle.remove()
        return out

class LatentJNet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, in_c))
        self.net = nn.Sequential(
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(32, in_c), nn.SiLU(), nn.Conv2d(in_c, in_c, 3, 1, 1)
        )
    def forward(self, h, t):
        t_vec = t.view(-1, 1).float() / 1000.0 
        t_feat = self.time_mlp(t_vec)[:, :, None, None]
        return torch.tanh(self.net(h + t_feat))

def compute_corrected_latent(h, u):
    B = h.shape[0]
    h_flat = h.reshape(B, -1); u_flat = u.reshape(B, -1)
    
    h_norm = torch.norm(h_flat, dim=1, keepdim=True) + 1e-6
    n_flat = h_flat / h_norm
    proj = (u_flat * n_flat).sum(dim=1, keepdim=True) * n_flat
    v_corr_flat = u_flat - proj
    
    v_norm = torch.norm(v_corr_flat, dim=1, keepdim=True) + 1e-6
    threshold = Config.energy_threshold * h_norm
    gating = torch.clamp(threshold / v_norm, max=1.0)
    
    v_corr = (v_corr_flat * gating).reshape(h.shape)
    return h + v_corr

def normalize_to_img(tensor):
    return (tensor.clamp(-1, 1) * 0.5 + 0.5)

# ==========================================
# 2. 核心扫描逻辑
# ==========================================
def run_scan():
    # 1. Load Models
    print(">>> Loading Models...")
    teacher = GoogleTeacherWrapper()
    latent_dim = teacher.get_latent_dim()
    
    # Load Student if available
    if os.path.exists(Config.path_student):
        student = LatentJNet(in_c=latent_dim).to(Config.device)
        student.load_state_dict(torch.load(Config.path_student))
        student.eval()
        print("✅ Student J-Net loaded.")
    else:
        print("⚠️ Student weights not found! Only running DDPM baseline.")
        student = None
        
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    
    # 2. Load Data
    transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))[:Config.num_samples]
    dataloader = DataLoader(Subset(dataset, indices), batch_size=Config.batch_size, shuffle=False)
    
    # 3. Init Metrics
    # FID 使用深层特征 (2048)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(Config.device)
    inc = InceptionScore(normalize=True).to(Config.device)
    
    print(">>> Caching Real Statistics...")
    real_cache = []
    for x_real, _ in tqdm(dataloader, desc="Real Stats"):
        x_real_norm = (x_real * 0.5 + 0.5).clamp(0, 1)
        real_cache.append(x_real_norm.to(Config.device))
        fid.update(x_real_norm.to(Config.device), real=True)
    
    history = {
        "steps": [], 
        "ddpm": {"mse": [], "fid": [], "is": []}, 
        "sddpm": {"mse": [], "fid": [], "is": []}
    }
    
    methods = ["DDPM", "SDDPM"] if student else ["DDPM"]
    
    # 4. Scanning Loop
    for t_val in Config.scan_steps:
        print(f"\n⚡ Scanning Step Count: {t_val} ...")
        
        for method in methods:
            # Reset metrics & Reinject Real stats
            fid.reset(); inc.reset()
            for x_real in real_cache: fid.update(x_real, real=True)
            
            total_mse = 0
            batch_count = 0
            
            for x_0, _ in tqdm(dataloader, desc=f"{method} {t_val}"):
                x_0 = x_0.to(Config.device)
                B = x_0.shape[0]
                
                # A. 严格加噪 (Forward to t_val steps away)
                # 注意：这里我们模拟的是 "只走一步，跨越 t_val 这么远"
                # 所以我们把图加噪到 t_val 时刻 (diffusers: 0~999)
                # 对应的 timestep index 是 t_val
                # 举例: t_val=50, 意味着我们处于第50步，要一步跨回0
                t_batch = torch.full((B,), t_val, device=Config.device).long()
                noise = torch.randn_like(x_0)
                x_t = scheduler.add_noise(x_0, noise, t_batch)
                
                # B. 一步去噪 (One-step Prediction)
                with torch.no_grad():
                    if method == "SDDPM":
                        # Latent Correction
                        h = teacher.get_mid_h(x_t, t_batch)
                        u = student(h, t_batch)
                        h_new, _ = compute_corrected_latent(h, u)
                        pred_noise = teacher.run_decoder(x_t, t_batch, h_new)
                    else:
                        # Standard DDPM
                        pred_noise = teacher.unet(x_t, t_batch).sample
                
                # DDIM/Euler One-step Update
                alpha_bar = scheduler.alphas_cumprod.to(Config.device)[t_batch].view(-1, 1, 1, 1)
                x_pred_0 = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
                
                # C. Post-process
                x_pred_0_norm = (x_pred_0 * 0.5 + 0.5).clamp(0, 1)
                x_0_norm = (x_0 * 0.5 + 0.5).clamp(0, 1)
                
                # MSE
                mse = F.mse_loss(x_pred_0_norm, x_0_norm).item()
                total_mse += mse
                batch_count += 1
                
                # Metrics
                fid.update(x_pred_0_norm, real=False)
                inc.update(x_pred_0_norm)
                
                # Viz (First batch only)
                if batch_count == 1 and method == "SDDPM": # 只画 SDDPM 的对比，或者分开画
                    pass 
            
            # Record
            avg_mse = total_mse / batch_count
            fid_score = fid.compute().item()
            is_score, _ = inc.compute(); is_score = is_score.item()
            
            print(f"   [{method}] MSE: {avg_mse:.5f} | FID: {fid_score:.2f} | IS: {is_score:.2f}")
            
            history[method.lower()]["mse"].append(avg_mse)
            history[method.lower()]["fid"].append(fid_score)
            history[method.lower()]["is"].append(is_score)
            
        history["steps"].append(t_val)

    plot_curves(history)
    print(f"\n✅ Scan Complete! Check {Config.save_dir}")

# ==========================================
# 5. 绘图
# ==========================================
def plot_curves(history):
    steps = history["steps"]
    metrics = ["mse", "fid", "is"]
    
    for m in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(steps, history["ddpm"][m], 'b-o', label='DDPM', linewidth=2)
        if len(history["sddpm"][m]) > 0:
            plt.plot(steps, history["sddpm"][m], 'r-^', label='SDDPM', linewidth=2)
            
        plt.title(f"{m.upper()} vs Step Size (t)")
        plt.xlabel("Step Size t (Distance to 0)")
        plt.ylabel(m.upper())
        plt.legend()
        plt.grid(True)
        if m != "is": plt.yscale("log") # Log scale for MSE/FID
        
        plt.savefig(f"{Config.save_dir}/curve_{m}.png")
        plt.close()
    
    with open(f"{Config.save_dir}/metrics.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    run_scan()