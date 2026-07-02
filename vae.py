import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    # 输入展平图像，输出mu和logvar
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__() # 继承父类初始化
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc1(x)) # h.shape = (B, hidden_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar # (B, latent_dim)
    
class Decoder(nn.Module):
    # 输入隐变量z，输出重建图像的 Bernoulli参数
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400, output_dim: int = 784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon
    
class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) # z.shape = (B, latent_dim)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    @torch.no_grad() # 把函数中所有计算的梯度追踪关掉
    def sample(self, num_samples:int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device = device)
        return self.decoder(z)
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return self.decoder(mu)

def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction = "sum") # 和kl_loss也是求和相匹配
    kl_loss = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
    total_loss = kl_loss + recon_loss
    return total_loss, recon_loss, kl_loss # 均为0维tensor
