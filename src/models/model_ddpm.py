import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
# Adapted from : https://github.com/dome272/Diffusion-Models-pytorch

class UNet(nn.Module):
    """
    A U-Net that **always emits the same number of channels it receives**.
    
    Parameters
    ----------
    channels     : sequence[int]
        The widths for every resolution level (encoder depth = len(channels)).
    grey_scale   : bool
        If True, the data have a single image channel; otherwise we assume four
        image channels (e.g. RGBA) and add the time channel → 5 total.
    """
    def __init__(self,
                 channels=(32, 64, 128, 256, 512, 1024, 1024),
                 grey_scale: bool = False):
        super().__init__()

        # ─── overall I/O channel counts ──────────────────────────────────────────
        self.nch = 2 if grey_scale else 5        # 1 / 4 img-ch + 1 time-ch
        chs      = list(channels)                # make a mutable copy

        # ─── Encoder (list-comprehension) ───────────────────────────────────────
        def _enc_block(in_ch, out_ch, first: bool):
            """First level: conv; deeper levels: pool → conv.  Ends in LogSigmoid."""
            layers = [] if first else [nn.MaxPool2d(2)]
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                       nn.LogSigmoid()]
            return nn.Sequential(*layers)

        self.encoder = nn.ModuleList(
            [_enc_block(self.nch if i == 0 else chs[i - 1], chs[i], i == 0)
             for i in range(len(chs))]
        )

        # ─── Decoder (list-comprehension) ───────────────────────────────────────
        def _dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.LogSigmoid()
            )

        # pure “×2” up-convs:
        decoder_ups = (
            [_dec_block(chs[-1], chs[-2])] +                              # first (no skip yet)
            [_dec_block(chs[-(i + 1)] * 2, chs[-(i + 2)])                 # after concat
             for i in range(1, len(chs) - 1)]
        )

        # final head that squashes back to `self.nch` channels
        decoder_tail = nn.Sequential(
            nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
            nn.LogSigmoid(),
            nn.Conv2d(chs[0], self.nch-1, kernel_size=3, padding=1)         # ← here
        )

        self.decoder = nn.ModuleList(decoder_ups + [decoder_tail])

    # ─── forward ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, nch-1, H, W) – raw image channels
        t : (B,) or (B,1)    – time step in [0,1]
        returns (B, nch-1, H, W)
        """
        B, _, H, W = x.shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # broadcast `t` into a single time channel and concat
        t_ch  = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        sig   = torch.cat([x, t_ch], dim=1)      # (B, self.nch, H, W)

        # ---- encode ----
        skips = []
        for i, enc in enumerate(self.encoder):
            sig = enc(sig)
            if i < len(self.encoder) - 1:        # keep all but bottleneck
                skips.append(sig)

        # ---- decode ----
        for i, dec in enumerate(self.decoder):
            if i == 0:                           # first up-conv – no skip yet
                sig = dec(sig)
            else:
                skip = skips[-i]                 # matching feature map
                sig  = torch.cat([sig, skip], dim=1)
                sig  = dec(sig)

        return sig


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100, p_unconditional=1.):
        """
        Initialiser en DDPM model.

        Parameters:
        network: [nn.Module]
            netværket til at bruge i diffusion processen.
        beta_1: [float]
            Støjet i frøste skridt af diffusion processen.
        beta_T: [float]
            Støjet i det sidste skridt af diffusion processen.
        T: [int]
            Maks antal skridt i diffusion processsen.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.p_unconditional = p_unconditional

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

        self.loss_criterion = nn.MSELoss(reduction='none')
    
    def negative_elbo(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        # 1) sample t uniformly and normalize
        t = torch.randint(1, self.T, (B, 1), device=self.alpha.device)
        normalized_t = (t + 1) / (self.T + 1)

        # 2) make noise of same shape
        epsilon = torch.randn_like(x)

        # 3) form x_t in image-space: broadcast alpha_cumprod[t] to (B,1,1,1)
        a_bar = self.alpha_cumprod[t].view(B, 1, 1, 1)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * epsilon

        # 4) predict noise with your UNet
        epsilon_params = self.network(x_t, normalized_t)

        # 5) per-sample squared error summed over (C,H,W)
        loss = self.loss_criterion(epsilon_params, epsilon).sum(dim=(1, 2, 3))

        return loss

    def sample(self, shape, keep_steps=False):
        """
        Sample fra modellen.

        Parameters:
        shape: [tuple]
            Dimensionerne af samples der skal genereres.
        Returns:
        [torch.Tensor]
            De genererede samples.
        """
        x_t = torch.randn(shape).to(self.alpha.device)

        if keep_steps:
            steps = [x_t]


        for t in range(self.T-1, -1, -1):
            if t > 1:
                z = torch.randn(shape).to(self.alpha.device)
            else:
                z = 0

            scale = 1 / torch.sqrt(self.alpha[t])
            epsilon_params = self.network(x_t, torch.tensor([(t + 1)/(self.T + 1)] * shape[0]).unsqueeze(1).to(self.alpha.device))

            parentheses = x_t - ((1 - self.alpha[t])/(torch.sqrt(1 - self.alpha_cumprod[t]))) * epsilon_params

            const = torch.sqrt(self.beta[t]) * z
            x_t = scale * parentheses + const
            
            if keep_steps:
                steps.append(x_t)
                
        if keep_steps:
            steps = torch.stack(steps, dim=1)
            return steps
        else:
            return x_t
    
    def forward(self, x):
        return self.negative_elbo(x).mean()



# === Example instantiation & forward pass ===
device = 'mps'

model = UNet(grey_scale=False).to(device)

# Print model parameter count
try:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")
except Exception as e:
    print(f"Could not calculate model parameters: {e}")

diffusion = DDPM(
    network=model,
    beta_1=1e-4,
    beta_T=0.02,
    T=1000,
    p_unconditional=1.0
).to(device)

# test a forward noising step:
x = torch.randn(32, 4, 64, 64, device=device)        # your batch
#x = x.view(x.shape[0], -1)  # flatten the image
loss = diffusion(x)
print(f"Loss: {loss.item()}")

# Test the sampling function
sampled_images = diffusion.sample((4, 4, 64, 64))
print(f"Sampled images shape: {sampled_images.shape}")  # Should be (32, T, 1, 64, 64)