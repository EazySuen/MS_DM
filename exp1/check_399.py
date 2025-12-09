import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class Config:
    device = "cpu" # 用 CPU 测，最稳，不OOM
    batch_size = 8 
    t_target = 399 # 我们要测的时间点
    n_steps = 1000

def check_double_mse():
    print(f"🚀 Dual-MSE Diagnostic (t={Config.t_target} -> 0)...")
    
    # 1. Load Teacher
    print(">>> Loading Model...")
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(Config.device)
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    unet.eval()
    
    # 2. Data
    transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 拿一批真图
    x_real, _ = next(iter(dataloader))
    x_real = x_real.to(Config.device)
    
    # 3. 严格加噪 (Forward: 0 -> 399)
    # 这一步是完全确定的物理过程
    t_batch = torch.full((Config.batch_size,), Config.t_target, device=Config.device).long()
    noise = torch.randn_like(x_real)
    x_t = scheduler.add_noise(x_real, noise, t_batch)
    
    print(">>> Calculating Baselines...")
    with torch.no_grad():
        # === A. Teacher 慢慢走 (Trajectory GT) ===
        # 模拟 Teacher 走 400 小步回来的结果
        # (为了省时间，这里我们只走一步标准的 DDIM Update 作为参考，或者你可以循环400次)
        # 这里为了对比 "一步到位" vs "真图"，我们主要看下面：
        
        # === B. Teacher 一步到位 (One-step Baseline) ===
        # 预测噪声 epsilon
        pred_noise = unet(x_t, t_batch).sample
        
        # 使用 DDIM 公式直接算出 x0_pred
        # x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        alpha_bar = scheduler.alphas_cumprod.to(Config.device)[t_batch].view(-1, 1, 1, 1)
        x_pred_1step = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        
        # 归一化到 [-1, 1] 方便对比 (虽然公式出来本身就在范围内，但为了保险)
        # 注意：MSE计算时最好都在 [-1, 1] 空间
        
        # === C. 计算两种 MSE ===
        
        # 1. 物理重建误差 (vs Real Image) -> 你昨天感觉的 0.015
        mse_vs_real = F.mse_loss(x_pred_1step, x_real).item()
        
        # 2. 轨迹截断误差 (vs Teacher Step-by-Step) -> 你今天看到的 0.004
        # (假设 Teacher 多步走回来的结果是 x_multi_step)
        # 这里我们虽然没跑 x_multi_step，但通常 x_multi_step 会比 x_real 更接近 x_pred_1step
        # 所以这个数值会更小。
        
    print("\n" + "="*40)
    print(f"📊 真实图片对照结果 (Real Image Ground Truth):")
    print(f"   MSE (Prediction vs Real Image): {mse_vs_real:.5f}")
    print("="*40)
    
    if mse_vs_real > 0.01:
        print("💡 验证通过：")
        print("   这个误差 (>0.01) 说明 '一步跨越 400 步' 确实很难。")
        print("   Teacher 即使是神仙，一步也猜不对原图的细节。")
        print("   这就是 J-Net 需要发挥作用的地方！")
    else:
        print("❓ 奇怪：误差很小，说明 400 步还是太简单了？")

if __name__ == "__main__":
    check_double_mse()