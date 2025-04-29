import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
# Adapted from : https://github.com/dome272/Diffusion-Models-pytorch


class Diffusion:
    def __init__(
        self,
        T=500,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        img_channels=1,
        device="cuda"
    ):
        """
        T : total diffusion steps
        beta_start: start of beta schedule
        beta_end: end of beta schedule
        img_size: spatial size of the image
        img_channels: number of channels in the image
        """
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.betas = self.get_betas('linear').to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T)
        else:
            raise NotImplementedError

    def q_sample(self, x, t):
        sqrt_ab = torch.sqrt(self.alphas_bar[t])[:, None, None, None]
        sqrt_omb = torch.sqrt(1 - self.alphas_bar[t])[:, None, None, None]
        noise = torch.randn_like(x)
        x_t = sqrt_ab * x + sqrt_omb * noise
        return x_t, noise

    def p_mean_std(self, model, x_t, t):
        alpha = self.alphas[t][:, None, None, None]
        alpha_bar = self.alphas_bar[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]

        pred_noise = model(x_t, t)
        mean = (1. / torch.sqrt(alpha)) * (
            x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise
        )
        std = torch.sqrt(beta)
        return mean, std

    def p_sample(self, model, x_t, t):
        mean, std = self.p_mean_std(model, x_t, t)
        # no noise at t=1
        noise = torch.stack([
            torch.randn_like(x_t[i]) if t[i] > 1 else torch.zeros_like(x_t[i])
            for i in range(x_t.shape[0])
        ])
        return mean + std * noise

    def p_sample_loop(self, model, batch_size, timesteps_to_save=None):
        logging.info(f"Sampling {batch_size} new images....")
        model.eval()
        intermediates = [] if timesteps_to_save else None

        with torch.no_grad():
            x = torch.randn(batch_size, self.img_channels,
                            self.img_size, self.img_size, device=self.device)
            for i in tqdm(reversed(range(1, self.T)), total=self.T-1):
                t = torch.full((batch_size,), i, dtype=torch.long,
                               device=self.device)
                x = self.p_sample(model, x, t)

                if timesteps_to_save and i in timesteps_to_save:
                    xi = (x.clamp(-1,1)+1)/2 * 255
                    intermediates.append(xi.type(torch.uint8))

        model.train()
        x = (x.clamp(-1,1)+1)/2 * 255
        x = x.type(torch.uint8)

        if intermediates is not None:
            intermediates.append(x)
            return x, intermediates
        return x

    def sample_timesteps(self, batch_size):
        return torch.randint(1, self.T, (batch_size,), device=self.device)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, img_size=64, c_in=1, c_out=1,
                 time_dim=256, device="cpu", channels=32):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # now accepts 1->32 channels
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2, emb_dim=time_dim)
        self.sa1   = SelfAttention(channels*2, img_size//2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        self.sa2   = SelfAttention(channels*4, img_size//4)
        self.down3 = Down(channels*4, channels*4, emb_dim=time_dim)
        self.sa3   = SelfAttention(channels*4, img_size//8)

        self.bot1  = DoubleConv(channels*4, channels*8)
        self.bot2  = DoubleConv(channels*8, channels*8)
        self.bot3  = DoubleConv(channels*8, channels*4)

        self.up1   = Up(channels*8, channels*2, emb_dim=time_dim)
        self.sa4   = SelfAttention(channels*2, img_size//4)
        self.up2   = Up(channels*4, channels, emb_dim=time_dim)
        self.sa5   = SelfAttention(channels, img_size//2)
        self.up3   = Up(channels*2, channels, emb_dim=time_dim)
        self.sa6   = SelfAttention(channels, img_size)

        self.outc  = nn.Conv2d(channels, c_out, kernel_size=1)

    # … pos_encoding and forward as before …

    def pos_encoding(self, t, channels):
        # compute 1/10000^(i/channels) as exp(-log(10000)*i/channels)
        i = torch.arange(0, channels, 2, device=self.device).float()
        inv_freq = torch.exp(- i * (math.log(10000.0) / channels)).to(t.device)
        # shape of t is [batch, 1]? assume [B,1]
        args = t * inv_freq.unsqueeze(0)          # broadcast to [B, channels//2]
        pos_enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bot3(self.bot2(self.bot1(x4)))

        x = self.sa4(self.up1(x4, x3, t))
        x = self.sa5(self.up2(x, x2, t))
        x = self.sa6(self.up3(x, x1, t))
        return self.outc(x)


# === Example instantiation & forward pass ===
device = 'mps'

model = UNet(
    img_size=64,
    c_in=1,      # <-- single‐channel input
    c_out=1,     # <-- single‐channel output
    device=device
).to(device)

diffusion = Diffusion(
    T=500,
    beta_start=1e-4,
    beta_end=0.02,
    img_size=64,
    img_channels=1,  # <-- match your data
    device=device
)

# test a forward noising step:
x = torch.randn(32, 1, 64, 64, device=device)        # your batch
t = diffusion.sample_timesteps(32)                    # random timesteps
x_t, noise = diffusion.q_sample(x, t)
pred_noise = model(x_t, t)
print(f"pred_noise shape: {pred_noise.shape}")