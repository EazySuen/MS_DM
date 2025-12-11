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
    stiff_val = 400.0  
    
    # 实验设定
    total_time = 0.05
    teacher_steps = 1000 
    student_steps = 1    
    
    dt_teacher = total_time / teacher_steps
    dt_student = total_time / student_steps
    
    batch_size = 64
    train_steps = 800
    lr = 0.005 
    
    lambda_spectral = 0.1
    warmup_steps = 100 # 缩短 warmup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Teacher 代码不变) ...
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

# ==========================================
# Student: Implicit Structure (inverse operator)
# ==========================================
class ImplicitStructureStudent(nn.Module):
    def __init__(self, mode='full_rank'):
        super().__init__()
        self.mode = mode
        
        # 依然没有 Gamma！靠几何结构本身来缩放！
        print(f"[{mode}] initialized with IMPLICIT Geometric Structure (I-J)^-1")
        
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
        v_base = teacher_instance.get_velocity(x) # [B, D]
        J = self.get_J() # [D, D]
        
        # 结构: v_new = (I - J)^-1 * v_base
        # 等价于解方程: (I - J) * v_new = v_base
        
        I = torch.eye(Config.dim, device=x.device)
        M = I - J
        
        # 注意：v_base 是 [B, D]，我们需要对每个样本解方程
        # 但 M 是共享的，所以可以转置后用 solve
        # M @ v_new^T = v_base^T
        
        # 对于 512 维，直接 solve 是最稳的
        v_new_T = torch.linalg.solve(M, v_base.T)
        v_new = v_new_T.T
        
        return v_new

# ... (Spectral Norm 代码不变) ...
def compute_spectral_norm(model, x, num_iters=5):
    x.requires_grad_(True)
    v_pred = model(x, teacher)
    u = torch.randn_like(x)
    u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-6)
    for _ in range(num_iters):
        v_jvp = torch.autograd.grad(v_pred, x, grad_outputs=u, create_graph=True, retain_graph=True)[0]
        u = v_jvp / (torch.norm(v_jvp, dim=1, keepdim=True) + 1e-6)
    return v_pred, torch.norm(v_jvp, dim=1).mean()

# ==========================================
# Run Battle
# ==========================================
def run_battle():
    print("\n⚔️ Phase 3: Implicit Structure Test ((I-J)^-1)")
    
    models = {
        'Full Rank': ImplicitStructureStudent('full_rank').to(Config.device),
        'LoRA': ImplicitStructureStudent('lora').to(Config.device)
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
            
    return history, models

def visualize_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    for name, data in history.items():
        ax1.plot(data['mse'], label=name, linewidth=2)
    ax1.set_title("Implicit Structure MSE")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, data in history.items():
        ax2.plot(data['lip'], label=name, linewidth=2)
    ax2.set_title("System Lipschitz Constant")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.savefig("implicit_battle.png")
    print("✅ Saved to implicit_battle.png")
    
# Visualize Spectrum
def visualize_spectrum_implicit(models):
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        if 'Full' in name: J = model.W - model.W.T
        else: J = model.U @ model.V.T - model.V @ model.U.T
        s = torch.linalg.svdvals(J.detach().cpu())
        plt.plot(s.numpy(), label=name, marker='o', markersize=3)
    plt.title("Spectrum of J (Implicit Model)")
    plt.yscale('log')
    plt.xlim(-1, 50)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("implicit_spectrum.png")

if __name__ == "__main__":
    hist, trained_models = run_battle()
    visualize_results(hist)
    visualize_spectrum_implicit(trained_models)