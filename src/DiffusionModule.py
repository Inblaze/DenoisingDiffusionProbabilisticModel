import math
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(data_path, img_size=64, batchsize=128):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize([img_size, img_size]),
        torchvision.transforms.ToTensor(),  # [0,1]
        torchvision.transforms.Lambda(lambda k: (k * 2) - 1)  # [-1,1]
    ])
    dataset = torchvision.datasets.Flowers102(root=data_path, split='train', transform=trans, download=True)
    return DataLoader(dataset, batchsize, True, drop_last=True)


class DiffusionModule:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def linear_beta_scheduler(self, tot_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, tot_steps).to(self.device)

    def cosine_beta_scheduler(self, timesteps, s=0.008):
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0, max=0.999).to(self.device)

    def get_selected_vals(self, vals, t, x_shape):
        batch_size = x_shape[0]
        out = vals.gather(-1, t.to(self.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def diffusion_process(self, x_0, t, mode='cosine', tot_steps=1000):
        x_0 = torch.Tensor(x_0).to(self.device)
        epsilon = torch.randn_like(x_0).to(self.device)
        if mode == 'linear':
            beta = self.linear_beta_scheduler(tot_steps)
        else:
            beta = self.cosine_beta_scheduler(tot_steps)
        alpha = (1 - beta).to(self.device)
        alpha_bar = torch.cumprod(alpha, dim=0).to(self.device)
        selected_alpha_bar = self.get_selected_vals(alpha_bar, t, x_0.shape)
        sqrt_selected_alpha_bar = torch.sqrt(selected_alpha_bar).to(self.device)
        sqrt_one_minus_selected_alpha_bar = torch.sqrt(1 - selected_alpha_bar).to(self.device)
        return (sqrt_one_minus_selected_alpha_bar * epsilon + sqrt_selected_alpha_bar * x_0).to(self.device), epsilon

    def __call__(self, x_0, t, mode='cosine', tot_steps=1000):
        return self.diffusion_process(x_0, t, mode, tot_steps)


if __name__ == '__main__':
    dpath = '../datasets/Flower102'
    dloader = get_dataloader(dpath)
    log_path = '../logs/Flower102'
    logger = SummaryWriter(log_path)
    T = 300
    num_of_phases = 10
    step_size = int(T / num_of_phases)
    imgs = next(iter(dloader))[0]
    logger.add_images('Diffusion Process', imgs, 0)
    my_diffusion_module = DiffusionModule()
    itr_imgs = imgs
    for i in range(9, T + 1, step_size):
        itr_imgs, noise = my_diffusion_module(itr_imgs, torch.full((8,), i), 'cosine', T)
        logger.add_images('Diffusion Process', itr_imgs, i + 1)
    logger.close()
