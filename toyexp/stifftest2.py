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
    
    # === 1. 刚性设定：足够让显式积分爆炸 ===
    # dt_student = 0.02. Stability limit is lambda < 2/dt = 100.
    # We set lambda = 400. Explosion factor = 7x per step.
    stiff_val = 400.0  
    
    # 蒸馏设定
    total_time = 1.0
    teacher_steps = 2000 # dt = 0.0005 (Safe)
    student_steps = 50   # dt = 0.02   (Unstable for raw teacher)
    
    distill_ratio = teacher_steps // student_steps
    dt_teacher = total_time / teacher_steps
    dt_student = total_time / student_steps
    
    batch_size = 64
    train_steps = 600
    lr = 0.01
    
    # 正则化权重: 惩罚 Spectral Norm
    lambda_spectral = 0.05
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Teacher: The Stiff Linear Field
# ==========================================
class StiffTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        # 随机正交基
        q, _ = torch.linalg.qr(torch.randn(Config.dim, Config.dim))
        self.register_buffer('Q', q)
        
        # 特征值: 4个是400，其余是1
        eigs = torch.ones(Config.dim)
        eigs[:Config.stiff_rank] = Config.stiff_val
        self.register_buffer('H', q @ torch.diag(eigs) @ q.T)

    def get_velocity(self, x):
        return - (self.H @ x.T).T

    def integrate(self, x0, steps, dt):
        x = x0.clone()
        for _ in range(steps):
            v = self.get_velocity(x)
            x = x + v * dt
        return x

teacher = StiffTeacher().to(Config.device)

# ==========================================
# Phase 1: 验证显式积分会爆炸
# ==========================================
def check_explosion():
    print(f"\n💥 Phase 1: Checking Baseline Explosion (Stiffness={Config.stiff_val})")
    x0 = torch.randn(1, Config.dim).to(Config.device)
    
    # Ground Truth (Small Steps)
    x_gt = teacher.integrate(x0, Config.teacher_steps, Config.dt_teacher)
    
    # Naive Large Step (Should Explode)
    try:
        x_naive = teacher.integrate(x0, Config.student_steps, Config.dt_student)
        diff = torch.norm(x_gt - x_naive).item()
        print(f"   Naive Approach Error: {diff:.2e}")
        if diff > 1e3:
            print("   ==> RESULT: SYSTEM EXPLODED! (As expected)")
        else:
            print("   ==> WARNING: System is too stable. Increase stiffness.")
    except RuntimeError:
        print("   ==> RESULT: NUMERICAL OVERFLOW! (Perfect Explosion)")

# ==========================================
# Student Models
# ==========================================
class DistilledStudent(nn.Module):
    def __init__(self, mode='full_rank'):
        super().__init__()
        self.mode = mode
        
        # 初始阻尼 gamma
        # 物理直觉：大步长要走得稳，通常需要把 Teacher 巨大的瞬时速度缩小
        self.gamma = nn.Parameter(torch.tensor(0.1)) 
        
        if mode == 'full_rank':
            # Algorithm A: Full Rank J (Noisy Initialization)
            self.W = nn.Parameter(torch.randn(Config.dim, Config.dim) * 0.01)
        elif mode == 'lora':
            # Algorithm B: LoRA J (Clean Initialization)
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
        
        # v = gamma * (I + J) * v_base
        v_rot = (J @ v_base.T).T
        return self.gamma * (v_base + v_rot)

# ==========================================
# Power Iteration (Spectral Norm)
# ==========================================
def compute_spectral_norm(model, x, num_iters=5):
    # 估算 Student 系统的 Lipschitz 常数
    # 即 max || dv_student / dx ||
    x.requires_grad_(True)
    v_pred = model(x, teacher)
    
    u = torch.randn_like(x)
    u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-6)
    
    for _ in range(num_iters):
        v_jvp = torch.autograd.grad(v_pred, x, grad_outputs=u, create_graph=True, retain_graph=True)[0]
        norm = torch.norm(v_jvp, dim=1, keepdim=True) + 1e-6
        u = v_jvp / norm
        
    # v_jvp is roughly J^T * u
    return v_pred, norm.mean()

# ==========================================
# Phase 2: Distillation Battle
# ==========================================
def run_distillation():
    print("\n⚔️ Phase 2: Distillation Battle (Full Rank vs LoRA)")
    
    models = {
        'Full Rank': DistilledStudent('full_rank').to(Config.device),
        'LoRA': DistilledStudent('lora').to(Config.device)
    }
    
    opts = {name: optim.Adam(m.parameters(), lr=Config.lr) for name, m in models.items()}
    history = {name: {'mse': [], 'lip': []} for name in models}
    
    for step in range(Config.train_steps):
        x0 = torch.randn(Config.batch_size, Config.dim).to(Config.device)
        
        # 1. Ground Truth (Slow & Accurate)
        with torch.no_grad():
            x_target = teacher.integrate(x0, Config.distill_ratio, Config.dt_teacher)
            # Student 目标：一步到位
            v_target = (x_target - x0) / Config.dt_student
        
        for name, model in models.items():
            opts[name].zero_grad()
            
            # 2. Forward + Spectral Norm
            v_pred, lip_val = compute_spectral_norm(model, x0)
            
            # 3. Loss
            loss_mse = nn.functional.mse_loss(v_pred, v_target)
            loss_reg = Config.lambda_spectral * lip_val
            
            loss = loss_mse + loss_reg
            loss.backward()
            opts[name].step()
            
            history[name]['mse'].append(loss_mse.item())
            history[name]['lip'].append(lip_val.item())
            
        if step % 50 == 0:
            print(f"Step {step} | "
                  f"FR MSE: {history['Full Rank']['mse'][-1]:.2f} Lip: {history['Full Rank']['lip'][-1]:.1f} | "
                  f"LoRA MSE: {history['LoRA']['mse'][-1]:.2f} Lip: {history['LoRA']['lip'][-1]:.1f}")
            
    return history

def plot_battle_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE Plot
    ax1 = axes[0]
    for name, data in history.items():
        ax1.plot(data['mse'], label=name, alpha=0.8, linewidth=1.5)
    ax1.set_title("Distillation MSE (Target Matching)")
    ax1.set_xlabel("Steps")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lipschitz Plot
    ax2 = axes[1]
    for name, data in history.items():
        ax2.plot(data['lip'], label=name, alpha=0.8, linewidth=1.5)
    # 画出 Teacher 的原始刚性基准线
    ax2.axhline(y=Config.stiff_val, color='r', linestyle='--', alpha=0.5, label='Teacher Stiffness (400)')
    # 画出 稳定性阈值 (1/dt = 50)
    ax2.axhline(y=1.0/Config.dt_student, color='g', linestyle='--', alpha=0.5, label='Stability Limit (50)')
    
    ax2.set_title("System Spectral Norm (Lipschitz Constant)")
    ax2.set_xlabel("Steps")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("real_physics_battle.png")
    print("\n✅ Saved to real_physics_battle.png")

if __name__ == "__main__":
    check_explosion()
    hist = run_distillation()
    plot_battle_results(hist)