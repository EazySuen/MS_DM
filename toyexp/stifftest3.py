import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 150

class Config:
    dim = 512
    stiff_rank = 4
    
    # === 难度升级 ===
    # 刚性 400
    stiff_val = 400.0  
    
    # Student 步长极大，dt = 0.05
    # 稳定性阈值 = 1 / 0.05 = 20.0
    # 任务：将刚性从 400 压到 20 以下！
    total_time = 0.05 
    teacher_steps = 1000 # dt = 0.00005 (Safe)
    student_steps = 1    # dt = 0.05   (Extremely Unstable for Teacher)
    
    dt_teacher = total_time / teacher_steps
    dt_student = total_time / student_steps
    
    batch_size = 64
    train_steps = 800
    lr = 0.01
    
    # 正则化强度
    lambda_spectral = 0.1
    warmup_steps = 200
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Teacher 代码不变，略) ...
class StiffTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(Config.dim, Config.dim))
        self.register_buffer('Q', q)
        eigs = torch.ones(Config.dim)
        eigs[:Config.stiff_rank] = Config.stiff_val
        self.register_buffer('H', q @ torch.diag(eigs) @ q.T)

    def get_velocity(self, x):
        return - (self.H @ x.T).T

    def integrate(self, x0, steps, dt):
        x = x0.clone()
        for _ in range(steps):
            x = x + self.get_velocity(x) * dt
        return x

teacher = StiffTeacher().to(Config.device)

# ... (Check Explosion 修复 NaN) ...
def check_explosion():
    print(f"\n💥 Phase 1: Checking Baseline Explosion")
    print(f"   Stiffness: {Config.stiff_val}")
    print(f"   Stability Threshold: {1.0/Config.dt_student:.1f}")
    
    x0 = torch.randn(1, Config.dim).to(Config.device)
    x_gt = teacher.integrate(x0, Config.teacher_steps, Config.dt_teacher)
    
    try:
        x_naive = teacher.integrate(x0, Config.student_steps, Config.dt_student)
        diff = torch.norm(x_gt - x_naive).item()
        
        # 修复 NaN 判断
        if np.isnan(diff) or diff > 1e3:
            print(f"   Naive Error: {diff}")
            print("   ==> RESULT: SYSTEM EXPLODED! (As expected)")
        else:
            print(f"   Naive Error: {diff}")
            print("   ==> WARNING: System stable? Increase stiffness or dt.")
            
    except RuntimeError:
        print("   ==> RESULT: NUMERICAL OVERFLOW! (Perfect)")

# ... (Student 代码，Gamma 初始更小) ...
class DistilledStudent(nn.Module):
    def __init__(self, mode='full_rank'):
        super().__init__()
        self.mode = mode
        
        # 初始阻尼设为 0.01，让它从极度过阻尼开始学
        self.gamma = nn.Parameter(torch.tensor(0.01)) 
        
        if mode == 'full_rank':
            self.W = nn.Parameter(torch.randn(Config.dim, Config.dim) * 0.01)
        elif mode == 'lora':
            r = Config.stiff_rank 
            self.U = nn.Parameter(torch.randn(Config.dim, r) * 0.01)
            self.V = nn.Parameter(torch.randn(Config.dim, r) * 0.01)
            
    def get_J(self):
        if self.mode == 'full_rank':
            return self.W - self.W.T
        elif self.mode == 'lora':
            return self.U @ self.V.T - self.V @ self.U.T

    def forward(self, x, teacher_instance):
        v_base = teacher_instance.get_velocity(x)
        J = self.get_J()
        v_rot = (J @ v_base.T).T
        return self.gamma * (v_base + v_rot)

# ... (Spectral Norm 迭代增加到 10 次) ...
def compute_spectral_norm(model, x, num_iters=10):
    x.requires_grad_(True)
    v_pred = model(x, teacher)
    u = torch.randn_like(x)
    u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-6)
    
    for _ in range(num_iters):
        v_jvp = torch.autograd.grad(v_pred, x, grad_outputs=u, create_graph=True, retain_graph=True)[0]
        u = v_jvp / (torch.norm(v_jvp, dim=1, keepdim=True) + 1e-6)
        
    return v_pred, torch.norm(v_jvp, dim=1).mean()

# ... (Run Battle 不变) ...
def run_battle():
    print("\n⚔️ Phase 2: High-Pressure Distillation")
    
    models = {
        'Full Rank': DistilledStudent('full_rank').to(Config.device),
        'LoRA': DistilledStudent('lora').to(Config.device)
    }
    
    opts = {name: optim.Adam(m.parameters(), lr=Config.lr) for name, m in models.items()}
    history = {name: {'mse': [], 'lip': []} for name in models}
    
    for step in range(Config.train_steps):
        x0 = torch.randn(Config.batch_size, Config.dim).to(Config.device)
        
        with torch.no_grad():
            x_target = teacher.integrate(x0, Config.teacher_steps, Config.dt_teacher)
            v_target = (x_target - x0) / Config.dt_student
            
        for name, model in models.items():
            opts[name].zero_grad()
            
            v_pred, lip_val = compute_spectral_norm(model, x0)
            
            loss_mse = nn.functional.mse_loss(v_pred, v_target)
            
            # Warmup
            curr_lambda = 0.0 if step < Config.warmup_steps else Config.lambda_spectral
            
            loss = loss_mse + curr_lambda * lip_val
            loss.backward()
            opts[name].step()
            
            history[name]['mse'].append(loss_mse.item())
            history[name]['lip'].append(lip_val.item())
            
        if step % 100 == 0:
            print(f"Step {step} | "
                  f"FR MSE: {history['Full Rank']['mse'][-1]:.2f} Lip: {history['Full Rank']['lip'][-1]:.1f} | "
                  f"LoRA MSE: {history['LoRA']['mse'][-1]:.2f} Lip: {history['LoRA']['lip'][-1]:.1f}")
            
    return history

def plot_battle_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE
    ax1 = axes[0]
    for name, data in history.items():
        ax1.plot(data['mse'], label=name, alpha=0.8)
    ax1.set_title("MSE")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lipschitz
    ax2 = axes[1]
    for name, data in history.items():
        ax2.plot(data['lip'], label=name, alpha=0.8)
    
    # Threshold Line
    threshold = 1.0 / Config.dt_student
    ax2.axhline(y=threshold, color='g', linestyle='--', label=f'Stability Threshold ({threshold})')
    
    ax2.set_title("Lipschitz Constant")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("death_match_result.png")
    print("\n✅ Saved to death_match_result.png")

def visualize_spectrum(models):
    plt.figure(figsize=(10, 5))
    
    for name, model in models.items():
        # 获取 J 矩阵
        if 'Full' in name:
            J = model.W - model.W.T
        else:
            J = model.U @ model.V.T - model.V @ model.U.T
        
        # 计算奇异值
        # detach() 并转到 cpu
        J_np = J.detach().cpu()
        s = torch.linalg.svdvals(J_np)
        
        plt.plot(s.numpy(), label=name, linewidth=2, marker='o', markersize=3, alpha=0.7)
        


# 在 run_battle 结束后调用
# visualize_spectrum(models) 
# 注意：你需要把 models 从 run_battle 返回出来

if __name__ == "__main__":
    check_explosion()
    hist = run_battle()
    plot_battle_results(hist)

    plt.title("Singular Value Spectrum of Learned Matrix J")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 50) # 我们只看前50个，因为 LoRA 秩很低
    
    plt.savefig("spectrum_check.png")
    print("✅ Spectrum visualization saved.")