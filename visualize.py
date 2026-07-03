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

def save_latent_interpolation_grid(model, data_loader, device, epoch, out_dir = "report", n_steps = 10):
    # 检验隐空间连续
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    x, _ = next(iter(data_loader))
    x_a = x[0:1].to(device) # (1, 1, 28, 28)
    x_b = x[1:2].to(device)
    x_a_flat = x_a.view(1, -1)
    x_b_flat = x_b.view(1, -1)

    with torch.no_grad():
        mu_a, _ = model.encoder(x_a_flat)
        mu_b, _ = model.encoder(x_b_flat)
        alphas = torch.linspace(0, 1, steps=n_steps, device=device).view(-1, 1)

        z_interp = (1-alphas) * mu_a + alphas * mu_b # 利用广播机制 (n_steps, latent_dim)
        imgs = model.decoder(z_interp).view(n_steps, 1, 28, 28)

    save_image(imgs, os.path.join(out_dir, f"interpolation_epoch{epoch}.png"), nrow = n_steps)
    model.train()

def save_latent_traversal_grid(model, device, epoch, out_dir = "report", 
                               base_z = None, dims = None, n_steps = 7, traversal_range = 3.0):
    # 检验维度是否真正“解耦”
    os.makedirs(out_dir, exist_ok = True)
    model.eval()
    
    if base_z is None:
        base_z = torch.zeros(1, model.latent_dim, device=device)
    if dims is None:
        dims = range(model.latent_dim)

    values = torch.linspace(-traversal_range, traversal_range, steps=n_steps, device=device)
    rows = []

    with torch.no_grad():
        for dim in dims:
            z_batch = base_z.repeat(n_steps, 1) # (n_steps, latent_dim) 先复制基准点
            z_batch[:,dim] = values
            imgs = model.decoder(z_batch).view(n_steps, 1, 28, 28)
            rows.append(imgs)

    grid = torch.cat(rows, dim=0) # (n_steps * n_steps, 1, 28, 28)
    save_image(grid, os.path.join(out_dir, f"traversal_epoch{epoch}.png"), nrow = n_steps)
    model.train()