import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=1000):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

def timestep_embedding(t: torch.Tensor, dim:int, max_period: float=10000.0) -> torch.Tensor:
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.squeeze(1)
    assert t.ndim == 1, "t must be [B] or [B,1]"
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=device)) * torch.arange(0, half, device=device).float() / half
    )

    args = t[:, None] * freqs[None, :] * 2.0 * torch.pi
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class TimeMLP(nn.Module):
    def __init__(self, emb_dim: int, cond_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        return self.net(e_t)

class FiLMHead(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

class TimeFiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, dropout: float=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = FiLMHead(cond_dim, out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        gamma, beta = self.film(cond)
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.drop(h)
        return h

class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: List[int] = (512, 128),
        emb_dim: int = 64,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.cond_dim = cond

        self.time_mlp = TimeMLP(emb_dim, cond_dim)

        # encoder
        enc_blocks = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_blocks.append(TimeFiLMBlock(in_dim, h, cond_dim, dropout))
            in_dim = h
        self.encoder_blocks = nn.ModuleList(enc_blocks)
        self.to_latent = nn.Linear(in_dim, latent_dim)

        # decoder 
        dec_blocks = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_blocks.append(TimeFiLMBlock(in_dim, h, cond_dim, dropout))
            in_dim = h
        self.decoder_blocks = nn.ModuleList(dec_blocks)
        self.to_output = nn.Linear(in_dim, input_dim)

        nn.init.zeros_(self.to_output.weight)
        nn.init.zeros_(self.to_output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e_t = timestep_embedding(t, self.emb_dim)    # [B, emb_dim]
        cond = self.time_mlp(e_t)                    # [B, cond_dim]

        h = x
        for block in self.encoder_blocks:
            h = block(h, cond)
        z = self.to_latent(h)                        # [B, latent]

        h = z
        for block in self.decoder_blocks:
            h = block(h, cond)
        x_hat = self.to_output(h)                    # [B, D]
        return x_hat, z

