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
    lr = 0.001 # 降低学习率，因为没有 gamma 保护，容易炸
    
    lambda_spectral = 0.1
    warmup_steps = 200
    
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
# Student: 纯粹的结构 (No Gamma!)
# ==========================================
class PureStructureStudent(nn.Module):
    def __init__(self, mode='full_rank'):
        super().__init__()
        self.mode = mode
        
        # ❌ 删除 Gamma！
        # self.gamma = nn.Parameter(torch.tensor(0.01)) 
        print(f"[{mode}] initialized WITHOUT Gamma scaling.")
        
        if mode == 'full_rank':
            self.W = nn.Parameter(torch.randn(Config.dim, Config.dim) * 0.001) # 初始化更小一点
        elif mode == 'lora':
            r = Config.stiff_rank 
            self.U = nn.Parameter(torch.randn(Config.dim, r) * 0.001)
            self.V = nn.Parameter(torch.randn(Config.dim, r) * 0.001)
            
    def get_J(self):
        if self.mode == 'full_rank':
            return self.W - self.W.T
        elif self.mode == 'lora':
            return self.U @ self.V.T - self.V @ self.U.T

    def forward(self, x, teacher_instance):
        v_base = teacher_instance.get_velocity(x)
        J = self.get_J()
        
        # 纯粹的物理结构: v_new = v_base + J * v_base
        # 没有任何人为的缩小！
        v_rot = (J @ v_base.T).T
        
        return v_base + v_rot

# ... (Spectral Norm 代码不变) ...
def compute_spectral_norm(model, x, num_iters=10):
    x.requires_grad_(True)
    v_pred = model(x, teacher)
    u = torch.randn_like(x)
    u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-6)
    for _ in range(num_iters):
        v_jvp = torch.autograd.grad(v_pred, x, grad_outputs=u, create_graph=True, retain_graph=True)[0]
        u = v_jvp / (torch.norm(v_jvp, dim=1, keepdim=True) + 1e-6)
    return v_pred, torch.norm(v_jvp, dim=1).mean()

# ==========================================
# Run Battle: The "Naked" Truth
# ==========================================
def run_battle():
    print("\n⚔️ Phase 2: The 'No-Gamma' Structure Test")
    
    models = {
        'Full Rank': PureStructureStudent('full_rank').to(Config.device),
        'LoRA': PureStructureStudent('lora').to(Config.device)
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

def visualize_spectrum_fixed(models):
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        if 'Full' in name:
            J = model.W - model.W.T
        else:
            J = model.U @ model.V.T - model.V @ model.U.T
        s = torch.linalg.svdvals(J.detach().cpu())
        plt.plot(s.numpy(), label=name, linewidth=2, marker='o', markersize=3, alpha=0.8)
    
    plt.title("Singular Value Spectrum of J (No Gamma Scaling)")
    plt.yscale('log')
    plt.ylim(1e-4, 100) 
    plt.xlim(-1, 30) 
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("spectrum_no_gamma.png")
    print("✅ Saved to spectrum_no_gamma.png")

if __name__ == "__main__":
    hist, trained_models = run_battle()
    visualize_spectrum_fixed(trained_models)