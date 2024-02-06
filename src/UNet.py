import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

class TimestepEmbedding(nn.Module):
    def __init__(self, model_ch:int, out_ch:int):
        super().__init__()
        assert model_ch % 2 == 0
        self.model_ch, self.out_ch = model_ch, out_ch
        self.timesteps_emb_seq = nn.Sequential(
            nn.Linear(model_ch, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        """
        :param t: 1-D Tensor of N indices, one per batch
        """
        half_ch = self.model_ch // 2
        emb = torch.exp(-torch.log(torch.Tensor([10000])) * torch.arange(half_ch, dtype=torch.float32) / (half_ch - 1)).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return self.timesteps_emb_seq(emb)

class ConditionalEmbedding(nn.Module):
    def __init__(self, model_ch:int, num_lables:int, out_ch:int):
        super().__init__()
        self.model_ch = model_ch
        self.out_ch = out_ch
        self.num_lables = num_lables
        self.conditional_emb_seq = nn.Sequential(
            nn.Embedding(num_lables+1, model_ch, padding_idx=0),
            nn.Linear(model_ch, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
    
    def forward(self, c:torch.Tensor) -> torch.Tensor:
        """
        :param c: 1-D Tensor of N indices, one per batch
        """
        return self.conditional_emb_seq(c) 

class DownSample(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, use_conv=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_conv = use_conv
        if use_conv:
            self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) does NOT match.'
        return self.layer(x)    

class UpSample(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, scale_factor=2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return self.conv(x)
    
class BlockWithEmbedment(nn.Module):
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        abstract method
        """

class MySequential(nn.Sequential, BlockWithEmbedment):
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, BlockWithEmbedment):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x

class ResidualBlock(BlockWithEmbedment):
    def __init__(self, in_ch:int, out_ch:int, temb_ch:int, cemb_ch:int, droprate=0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.temb_ch, self.cemb_ch = temb_ch, cemb_ch
        self.droprate = droprate
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_ch, out_ch)
        )
        self.cemb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cemb_ch, out_ch)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        latent += self.cemb_proj(cemb)[:, :, None, None]
        latent = self.block2(latent)
        return latent + self.residual(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.proj_q = nn.Conv2d(channels, channels, 1)
        self.proj_k = nn.Conv2d(channels, channels, 1)
        self.proj_v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.proj_q(h), self.proj_k(h), self.proj_v(h)
        q = q.permute(0, 2, 3, 1).reshape(N, H*W, C)
        k = k.reshape(N, C, H*W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)
        v = v.permute(0, 2, 3, 1).reshape(N, H*W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [N, H*W, C]
        h = h.reshape(N, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h

class UNet(nn.Module):
    def __init__(self, in_ch=3, model_ch=64, out_ch=3, ch_mul=[1,2,4,8], num_res_blocks=2, cdim=10, num_lables=10, use_conv=True, droprate=0.1, dtype=torch.float32):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.model_ch = model_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        self.dtype = dtype
        tc_dim = model_ch * 4
        self.temb_block = TimestepEmbedding(model_ch, tc_dim)
        self.cemb_block = ConditionalEmbedding(cdim, num_lables, tc_dim)
        #Initial Conv
        self.downblocks = nn.ModuleList([
            MySequential(nn.Conv2d(in_ch, model_ch, kernel_size=3, padding=1))
        ])
        # DownBlocks
        now_ch = ch_mul[0] * model_ch
        chs = [now_ch]
        for i, mul in enumerate(ch_mul):
            next_ch = model_ch * mul
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(now_ch, next_ch, tc_dim, tc_dim, droprate),
                    AttentionBlock(next_ch)
                ]
                now_ch = next_ch
                self.downblocks.append(MySequential(*layers))
                chs.append(now_ch)
            if i != len(ch_mul) - 1:
            # there are only (len(ch_mul)-1) downsamples
                self.downblocks.append(MySequential(DownSample(now_ch, now_ch, use_conv)))
                chs.append(now_ch)
        # MiddleBlocks
        self.middle_blocks = MySequential(
            ResidualBlock(now_ch, now_ch, tc_dim, tc_dim, droprate),
            AttentionBlock(now_ch),
            ResidualBlock(now_ch, now_ch, tc_dim, tc_dim, droprate)
        )
        # UpBlocks
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(ch_mul))[::-1]:
            next_ch = model_ch * mul
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(now_ch+chs.pop(), next_ch, tc_dim, tc_dim, droprate),
                    AttentionBlock(next_ch)
                ]
                self.upblocks.append(MySequential(*layers))
                now_ch = next_ch
            # ResBlock + AttnBlock + UpSample
            layers = [
                ResidualBlock(now_ch+chs.pop(), next_ch, tc_dim, tc_dim, droprate),
                AttentionBlock(next_ch)
            ]
            now_ch = next_ch
            if i != 0:
                layers.append(UpSample(now_ch, now_ch))
            self.upblocks.append(MySequential(*layers))
        # Output Block
        self.output_block = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x:torch.Tensor, t:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        temb, cemb = self.temb_block(t), self.cemb_block(c)
        latent = x.type(self.dtype)
        latents = []
        for block in self.downblocks:
            latent = block(latent, temb, cemb)
            latents.append(latent)
        latent = self.middle_blocks(latent, temb, cemb)
        for block in self.upblocks:
            latent = torch.cat([latent, latents.pop()], dim=1)
            latent = block(latent, temb, cemb)
        latent = latent.type(self.dtype)
        return self.output_block(latent)

if __name__ == '__main__':
    net = UNet()
    for name, params in net.named_parameters():
        print(name, ': ', params.size())
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'\nTotal: {total_num}\nTrainable: {trainable_num}')