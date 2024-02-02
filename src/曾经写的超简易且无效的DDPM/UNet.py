import math
import numpy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embed_channels):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embed_channels = embed_channels

    def forward(self, t):
        two_i = numpy.array([k if k % 2 == 0 else k - 1 for k in range(self.embed_channels)])
        div_term = torch.exp(torch.Tensor(-math.log(10000.0) * two_i / (self.embed_channels - 1))).to(self.device)
        indep_variable = t[:, None] * div_term[None, :]
        sin_of_var = torch.sin(indep_variable).to(self.device)
        cos_of_var = torch.cos(indep_variable).to(self.device)
        return torch.Tensor([[sin_of_var[i][j] if j % 2 == 0 else cos_of_var[i][j]
                              for j in range(sin_of_var.shape[1])] for i in range(sin_of_var.shape[0])]).to(self.device)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, embed_channels):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.emb = SinusoidalPositionEmbeddings(embed_channels)
        self.linear = nn.Linear(embed_channels, out_channels)
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.bnorm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x, t):
        x = x.to(torch.float)
        h = self.relu1(self.bnorm1(self.conv1(x)))
        embeddings = self.relu0(self.linear(self.emb(t)))
        embeddings = torch.unsqueeze(torch.unsqueeze(embeddings, -1), -1)
        h = h + embeddings
        h = self.relu2(self.bnorm2(self.conv2(h)))
        down_h = self.relu3(self.bnorm3(self.down(h)))
        return h, down_h


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.emb_ch = emb_channels
        double_out_ch = out_channels * 2
        self.emb = SinusoidalPositionEmbeddings(emb_channels)
        self.linear = nn.Linear(emb_channels, double_out_ch)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, double_out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(double_out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(double_out_ch, double_out_ch, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(double_out_ch)
        self.relu2 = nn.ReLU()
        self.up = nn.ConvTranspose2d(double_out_ch, out_channels, kernel_size=2, stride=2)
        self.bnorm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x, t):
        h = self.relu1(self.bnorm1(self.conv1(x)))
        embeddings = self.relu0(self.linear(self.emb(t)))
        embeddings = torch.unsqueeze(torch.unsqueeze(embeddings, -1), -1)
        h = h + embeddings
        h = self.relu2(self.bnorm2(self.conv2(h)))
        h = self.relu3(self.bnorm3(self.up(h)))
        return h


class SimpleUnet(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.img_ch = img_channels
        # self.down1 = DownSample(img_channels, 64, 64)  # img_channels -> 64 -> 64
        # self.donw2 = DownSample(64, 128, 128)  # 64 -> 128 -> 128
        # self.down3 = DownSample(128, 256, 256)  # 128 -> 256 -> 256
        # self.down4 = DownSample(256, 512, 512)  # 256 -> 512 ->512
        self.downs = nn.ModuleList([
            DownSample(img_channels, 64, 64),  # img_channels -> 64 -> 64
            DownSample(64, 128, 128),  # 64 -> 128 -> 128
            DownSample(128, 256, 256),  # 128 -> 256 -> 256
            DownSample(256, 512, 512)  # 256 -> 512 ->512
        ])
        self.up1 = UpSample(512, 512, 512)  # 512 -> 1024 -> 512
        # self.up2 = UpSample(1024, 256, 1024)  # 512 + 512 -> 512 ->256
        # self.up3 = UpSample(512, 128, 512)  # 256 + 256 -> 256 -> 128
        # self.up4 = UpSample(256, 64, 256)  # 128 + 128 -> 128 -> 64
        # self.out = nn.Sequential(  # 64 + 64 -> 64 -> img_channels
        #     nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        #     nn.Conv2d(64, img_channels, 3, padding=1)
        # )
        self.ups = nn.ModuleList([
            UpSample(1024, 256, 1024),  # 512 + 512 -> 512 ->256
            UpSample(512, 128, 512),  # 256 + 256 -> 256 -> 128
            UpSample(256, 64, 256),  # 128 + 128 -> 128 -> 64
        ])
        self.output = nn.Sequential(  # 64 + 64 -> 64 -> 64 -> img_channels
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1), nn.BatchNorm2d(img_channels)
        )

    def forward(self, x, t):
        x = x.to(self.device)
        t = torch.Tensor(t).to(self.device)
        residual_x = []
        for down in self.downs:
            rx, x = down(x, t)
            residual_x.append(rx)

        x = self.up1(x, t)
        for up in self.ups:
            rx = residual_x.pop()
            x = torch.cat((x, rx), dim=1)
            x = up(x, t)
        rx = residual_x.pop()
        x = torch.cat((x, rx), dim=1)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = SimpleUnet(3)
    print(model)
    # logger = SummaryWriter('../logs/network')
    # logger.add_graph(model, [torch.randn([8, 3, 64, 64]), torch.full((8,), 6)])
    # logger.close()