import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    
    # 核心目标：生成 t=0 到 t=400 的轨迹
    distill_skip = 400 
    n_steps = 1000
    
    num_batches = 200 
    # 路径改为 400 专用，防止和 500 的数据混淆
    save_dir = "./cifar_cache_ode_inversion_400"

os.makedirs(Config.save_dir, exist_ok=True)

def prepare_ode_data():
    print(f"🚀 Starting ODE Inversion Data Generation (0 -> {Config.distill_skip})...")
    
    # 1. Load Models
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(Config.device)
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    unet.eval()
    
    # 2. Load Real Data
    transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    
    print(f"Generating {Config.num_batches} batches via ODE Forward Integration...")
    
    # 终点 timestep
    T_end = Config.distill_skip
    
    for batch_idx in tqdm(range(Config.num_batches)):
        try:
            x_0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_0, _ = next(data_iter)
            
        x_curr = x_0.to(Config.device)
        
        # 缓存起点 (t=0)
        x_at_0 = x_curr.clone() 
        
        # === ODE Forward Integration (0 -> 400) ===
        # 循环从 0 跑到 400
        for t in range(T_end): 
            t_batch = torch.full((x_curr.shape[0],), t, device=Config.device).long()
            
            # 1. 预测噪声 (用于 DDIM 公式)
            noise_pred = unet(x_curr, t_batch).sample
            
            # 2. 获取下一时刻的 Alpha Bar (t+1)
            # DDIM Forward Update: x_{t+1} = sqrt(alpha_bar_{t+1}) * pred_x0 + sqrt(1 - alpha_bar_{t+1}) * pred_noise
            
            # 预测 x0 (clean)
            alpha_bar_t = scheduler.alphas_cumprod[t]
            sqrt_recip_alpha_bar = (1.0 / alpha_bar_t).sqrt()
            sqrt_recip_m1 = (1.0 / alpha_bar_t - 1.0).sqrt()
            pred_x0 = sqrt_recip_alpha_bar * x_curr - sqrt_recip_m1 * noise_pred
            
            # 3. 重新混合 (生成 t+1 时刻的 x)
            if t < T_end - 1:
                # 使用标准的DDIM公式，保证了轨迹的数学一致性
                alpha_bar_next = scheduler.alphas_cumprod[t+1]
                x_curr = alpha_bar_next.sqrt() * pred_x0 + (1 - alpha_bar_next).sqrt() * noise_pred
            else:
                # 最后一个 step (t=399) 结束后，x_curr 就是 x_400
                alpha_bar_end = scheduler.alphas_cumprod[T_end] if T_end < Config.n_steps else torch.tensor(0.0)
                x_curr = alpha_bar_end.sqrt() * pred_x0 + (1 - alpha_bar_end).sqrt() * noise_pred
                
                # 记录终点 x_at_end
                x_at_end = x_curr.clone()

        # === 保存数据 ===
        # 我们只关心从 t=400 (噪声) 一步去噪到 t=0 (真图) 的任务
        # Pair: x_t (t=400) -> x_target (t=0)
        save_path = f"{Config.save_dir}/batch_{batch_idx}.pt"
        torch.save({
            "x_t": x_at_end.cpu(),          # 输入 (t=400 的噪声)
            "t": torch.full((Config.batch_size,), T_end).long(), # 时间步 (400)
            "x_target": x_at_0.cpu()        # 目标 (t=0 的真图)
        }, save_path)

    print(f"✅ ODE Data Generation Complete! Saved to {Config.save_dir}")

    # ... (保存数据代码) ...
        
    # 强制清空 CUDA 缓存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    prepare_ode_data()