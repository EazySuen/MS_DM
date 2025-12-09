import os
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from utils.networkHelper import *

from noisePredictModels.Unet.UNet import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from diffusionModels.simpleDiffusion.simpleDiffusion import DiffusionModel


# 数据集加载
data_root_path = "./dataset/"
if not os.path.exists(data_root_path):
    os.makedirs(data_root_path)

imagenet_data = torchvision.datasets.FashionMNIST(data_root_path, train=True, download=True, transform=transforms.ToTensor())

image_size = 28
channels = 1
batch_size = 128

data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4,)

denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)

timesteps = 1000
schedule_name = "linear_beta_schedule"
DDPM = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.0001,
                      beta_end=0.02,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(DDPM.parameters(), lr=1e-3)
epoches = 20

Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)

# 训练参数设置
root_path = "./saved_train_models"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults, timesteps, schedule_name)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


# 训练好的模型加载，如果模型是已经训练好的，则可以将下面两行代码取消注释
best_model_path = saved_path + '/' + 'BestModel.pth'
DDPM.load_state_dict(torch.load(best_model_path))

# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
# DDPM = Trainer(DDPM, model_save_path=saved_path)

# 采样:sample 64 images
samples = DDPM(mode="generate", image_size=image_size, batch_size=64, channels=channels)

# 随机挑一张显示
random_index = 1
generate_image = samples[-1][random_index].reshape(channels, image_size, image_size)
figtest = reverse_transform(torch.from_numpy(generate_image))



# 在很多服务器/远程环境没有桌面环境，Image.show() 无法弹窗展示图片。
# 改为保存到文件夹，并在本地有 DISPLAY 时才尝试 show()
output_dir = os.path.join(saved_path, 'generated_images') if 'saved_path' in locals() else './generated_images'
os.makedirs(output_dir, exist_ok=True)

# 保存单张随机挑选的图片
single_path = os.path.join(output_dir, f'sample_{random_index}.png')
figtest.save(single_path)
print(f'已将随机生成的图片保存至: {single_path}')

# 另外保存全部生成图片（最后一步的 batch）便于批量查看
final_imgs = samples[-1]  # numpy array shape (batch, C, H, W)
import torchvision.utils as vutils
final_tensor = torch.from_numpy(final_imgs)
try:
    # 归一化到0-1再保存网格
    grid = vutils.make_grid(final_tensor, nrow=8, normalize=True, scale_each=True)
    # grid is C x H x W tensor in range 0-1; convert to PIL and save
    grid_img = reverse_transform(grid)
    grid_path = os.path.join(output_dir, 'grid.png')
    grid_img.save(grid_path)
    print(f'已将图片网格保存至: {grid_path}')
except Exception as e:
    print('生成图片网格失败:', e)

# 如果有桌面环境，仍然尝试打开图片
if os.environ.get('DISPLAY'):
    try:
        figtest.show()
    except Exception:
        pass


