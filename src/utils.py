import numpy as np
import math
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from MyDataset import MyDataset
from UNet import UNet

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# def get_betas(schedule_mode='linear',
#               tot_timesteps=1000,
#               max_beta=0.999) -> np.ndarray:
#     beta_start = 0.0001
#     beta_end = 0.02
#     if schedule_mode == 'linear':
#         return np.linspace(beta_start, beta_end, tot_timesteps, dtype=np.float64)
#     elif schedule_mode == 'cosine':
#         s = 0.008
#         func_cos = lambda t: math.cos((math.pi/2) * (t / tot_timesteps + s) / (1 + s)) ** 2
#         betas = []
#         for i in range(tot_timesteps):
#             tmp_t = func_cos(i)
#             tmp_t_plus_1 = func_cos(i + 1)
#             betas.append(min(1 - tmp_t_plus_1 / tmp_t, max_beta))
#         return np.array(betas)
#     else:
#         raise NotImplementedError(f"unknown beta schedule: {schedule_mode}")
    
def get_cifar_dataloader(batchsize:int, dpath:str):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10(
        root=dpath,
        train=True,
        transform=trans,
        download=True
    )
    loader = DataLoader(
        dataset=train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    return loader

def get_agfw_dataloader(batchsize:int, dpath:str):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = MyDataset(
        root=dpath,
        transform=trans
    )
    loader = DataLoader(
        dataset=train_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    return loader

def get_model(T:int,
              num_labels:int,
              ch:int,
              img_size:int = 64,
              num_res_blocks:int = 2,
              dropout:float = 0.0):
    if img_size == 512:
        channel_mult = [0.5, 1, 1, 2, 2, 4, 4]
    elif img_size == 256:
        channel_mult = [1, 1, 2, 2, 4, 4]
    elif img_size == 128:
        channel_mult = [1, 1, 2, 2, 2]
    elif img_size == 64:
        channel_mult = [1, 2, 3, 4]
    elif img_size == 32:
        channel_mult = [1, 2, 2, 2]
    else:
            raise ValueError(f"unsupported image size: {img_size}")
    
    return UNet(T=T,
                num_labels=num_labels,
                ch=ch,
                ch_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout)
    

def str2bool(s:str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ['true', 'yes', 't', 'y', '1']:
        return True
    else:
        return False
