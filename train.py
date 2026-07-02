import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae import VAE, vae_loss

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 20
EPOCHS = 1
LR = 1e-3
DATA_DIR = './data'

def get_dataloader() -> DataLoader:
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root = DATA_DIR, train = True, download = True, transform = transform)
    return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)

def train():
    train_loader = get_dataloader()
    model = VAE(input_dim = 784, hidden_dim = 400, latent_dim = LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss_sum, recon_loss_sum, kl_loss_sum, n_samples = 0.0, 0.0, 0.0, 0

        for batch_idx, (x, _) in enumerate(train_loader): # train_loader中有图像数据和标签，VAE是无监督学习，所以将标签舍弃
            x = x.view(x.size(0), -1).to(DEVICE) 

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss_sum+=loss.item() # 把tensor转成float
            recon_loss_sum+=recon_loss.item()
            kl_loss_sum+=kl_loss.item()
            n_samples+=x.size(0)

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(x)} / {len(train_loader.dataset)}]")
                print(f'batch avg loss: {loss.item() / len(x):.4f}')
        
        print(f'====> Epoch {epoch} 平均 loss: {total_loss_sum / n_samples:.4f} '
              f'(recon: {recon_loss_sum / n_samples:.4f}, kl: {kl_loss_sum / n_samples:.4f})')
        
if __name__ == '__main__':
    train()
