import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
import csv
import os

from vae import VAE, vae_loss
from visualize import save_reconstruction_grid, save_sample_grid

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description = "Train a simple MLP VAE on MNIST")
    parser.add_argument("--epochs", type=int, default = 10, help = "训练轮数")
    parser.add_argument("--batch_size", type=int, default = 128)
    parser.add_argument("--latent_dim", type=int, default = 20)
    parser.add_argument("--hidden_dim", type=int, default = 400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default = "./data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--report_dir", type=str, default="./report")
    parser.add_argument("--log_interval", type=int, default=100, help="每多少个 batch 打印一次日志")
    return parser.parse_args()

def get_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root = data_dir, train = True, download = True, transform = transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle = True)

def train():
    args = parse_args()

    torch.manual_seed(args.seed) # 设置随机种子，保证实验可复现

    run_name = f"beta{args.beta}"
    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    report_dir = os.path.join(args.report_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    # 加上exist_ok=True参数，避免文件夹已存在时报错

    train_loader = get_dataloader(args.data_dir, args.batch_size)
    model = VAE(input_dim = 784, hidden_dim = 400, latent_dim = args.latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    csv_path = os.path.join(args.report_dir, "loss_log.csv") # 自动使用当前系统的正确分隔符
    with open(csv_path, "w", newline="") as f:
        # "w"表示写入模式，若文件已存在则覆盖；newline=""避免在Windows上写入多余的空行
        # with ... as f 是上下文管理器语法，能够自动关闭文件（即使不是正常结束）
        writer = csv.writer(f)
        # 创建一个和文件f绑定的写入器
        writer.writerow(["epoch", "beta", "total_loss", "recon_loss", "kl_loss"])

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss_sum, recon_loss_sum, kl_loss_sum, n_samples = 0.0, 0.0, 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader): # train_loader中有图像数据和标签，VAE是无监督学习，所以将标签舍弃
            # x 的形状：（B, 1, 28, 28）
            x = x.view(x.size(0), -1).to(DEVICE) 

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            _, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
            loss = recon_loss + args.beta * kl_loss
            loss.backward()
            optimizer.step()

            total_loss_sum+=loss.item() # 把tensor转成float
            recon_loss_sum+=recon_loss.item()
            kl_loss_sum+=kl_loss.item()
            n_samples+=x.size(0)

            if batch_idx % 100 == 0:
                print(f"[beta={args.beta}] Epoch {epoch} [{batch_idx * len(x)} / {len(train_loader.dataset)}]")
                print(f'batch avg loss: {loss.item() / len(x):.4f}')
        
        avg_total = total_loss_sum / n_samples
        avg_recon = recon_loss_sum / n_samples
        avg_kl = kl_loss_sum / n_samples
        print(f"====> [beta={args.beta}] Epoch {epoch} 平均 loss: {avg_total:.4f} "
              f"(recon: {avg_recon:.4f}, kl: {avg_kl:.4f})")
        
        with open(csv_path, "a", newline="") as f:
            # "a"表示追加模式
            writer = csv.writer(f)
            writer.writerow([epoch, avg_total, avg_recon, avg_kl])

        ckpt_path = os.path.join(ckpt_dir, f"vae_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)

        save_reconstruction_grid(model, train_loader, DEVICE, epoch, out_dir=args.report_dir)
        save_sample_grid(model, DEVICE, epoch, num_samples=64, out_dir=args.report_dir)
        
if __name__ == '__main__':
    train()
