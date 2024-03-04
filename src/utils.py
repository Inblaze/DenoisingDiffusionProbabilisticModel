import numpy as np
import math
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_betas(schedule_mode='linear',
              tot_timesteps=1000,
              max_beta=0.999) -> np.ndarray:
    beta_start = 0.0001
    beta_end = 0.02
    if schedule_mode == 'linear':
        return np.linspace(beta_start, beta_end, tot_timesteps, dtype=np.float64)
    elif schedule_mode == 'cosine':
        s = 0.008
        func_cos = lambda t: math.cos((math.pi/2) * (t / tot_timesteps + s) / (1 + s)) ** 2
        betas = []
        for i in range(tot_timesteps):
            tmp_t = func_cos(i)
            tmp_t_plus_1 = func_cos(i + 1)
            betas.append(min(1 - tmp_t_plus_1 / tmp_t, max_beta))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_mode}")
    
def get_dataloader(batchsize:int, dpath:str):
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

def str2bool(s:str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ['true', 'yes', 't', 'y', '1']:
        return True
    else:
        return False
