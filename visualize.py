import os
import torch
from torchvision.utils import save_image

def save_reconstruction_grid(model, data_loader, device, epoch, out_dir="report", n = 8):
    os.makedirs(out_dir, exist_ok = True)
    model.eval()

    x, _ = next(iter(data_loader)) 
    # 把 data_loader 转化为一个可迭代对象，并取出下一个元素（由于这是新迭代器，“下一个”就是“第一个”
    x = x[:n].to(device) # 只切片取前八张图
    x_flat = x.view(x.size(0), -1)

    x_recon = model.reconstruct(x_flat).view_as(x)
    # reconstruct 返回的是展平的图像，.view_as把它重新变回和x一样的形状
    comparison = torch.cat([x, x_recon]) # 默认沿着第0维拼接，得到 (2n, 1, 28, 28) 的 tensor

    save_image(comparison, os.path.join(out_dir, f"reconstruction_epoch{epoch}.png"), nrow = n)
    # 每一行放n张图
    model.train()

def save_sample_grid(model, device, epoch, num_samples = 64, out_dir = "report"):
    os.makedirs(out_dir, exist_ok = True)
    model.eval()

    # 直接从先验分布里取样丢进decoder
    samples = model.sample(num_samples, device).view(num_samples, 1, 28, 28)
    save_image(samples, os.path.join(out_dir, f"sample_epoch{epoch}.png"), nrow = 8)

    model.train()