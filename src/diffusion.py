import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def extract(arr:torch.Tensor,
            idx:torch.Tensor,
            x_shape:tuple,
            device:torch.device) -> torch.Tensor:
    chosen = torch.gather(arr, dim=0, index=idx).float().to(device)
    return chosen.view([idx.shape[0]] + (len(x_shape) - 1) * [1])

class Diffusion(nn.Module):
    def __init__(self,
                 model, 
                 betas:np.ndarray, 
                 w: torch.float,
                 device:torch.device):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.betas = torch.tensor(betas).to(device)
        self.T = len(self.betas)
        self.w = w
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:self.T]
        self.tilde_beta = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.p_coeff1 = torch.sqrt(1. / self.alphas)
        self.p_coeff2 = self.p_coeff1 * (1. - self.alphas) / self.sqrt_one_minus_alphas_bar
        self.var = torch.cat([self.tilde_beta[1:2], self.betas[1:]])

    def q_sample(self,
                 x_0:torch.Tensor,
                 t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # q(x_t|x_0)
            eps_t = torch.randn_like(x_0)
            x_t = extract(self.sqrt_alphas_bar, t, x_0.shape, self.device) * x_0 + \
                  extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape, self.device) * eps_t
            return x_t, eps_t 

    def p_mean_variance(self,
                        x_t:torch.Tensor,
                        t:torch.Tensor,
                        c:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        # p_{theta}(x_{t-1}|x_t)
        variance = extract(self.var, t, x_t.shape, self.device)
        eps_cond = self.model(x_t, t, c)
        eps_uncond = self.model(x_t, t, torch.zeros_like(c).to(self.device))
        eps = (1 + self.w) * eps_cond - self.w * eps_uncond
        mean = extract(self.p_coeff1, t, x_t.shape, self.device) * x_t - \
               extract(self.p_coeff2, t, x_t.shape, self.device) * eps
        return mean, variance
    
    def p_sample(self,
                 x_t:torch.Tensor,
                 t:torch.Tensor,
                 c:torch.Tensor) -> torch.Tensor:
        mean, variance = self.p_mean_variance(x_t, t, c)
        z = torch.randn_like(x_t)
        z[t <= 0] = 0.0
        x_t_prev = mean + torch.sqrt(variance) * z
        assert torch.isnan(x_t_prev).int().sum() == 0, "NaN in tensor."
        return x_t_prev
    
    def p_sample_loop(self,
                      x_shape:tuple,
                      c:torch.Tensor) -> torch.Tensor:
        x_t = torch.randn(x_shape, device=self.device)
        for time_step in tqdm(list(reversed(range(self.T))), desc='Sampling', leave=False, dynamic_ncols=True):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            x_t = self.p_sample(x_t, t, c)
            assert torch.isnan(x_t).int().sum() == 0, "NaN in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def ddim_p_mean_variance(self,
                             x_t:torch.Tensor,
                             t:torch.Tensor,
                             t_prev:torch.Tensor,
                             eta:float,
                             c:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the mean and the variance of p_{theta}(x_{t-1}|x_{t}) in DDIM
        """
        # variance
        alphas_bar_t = extract(self.alphas_bar, t, x_t.shape, self.device)
        alphas_bar_prevt = extract(self.alphas_bar_prev, t_prev+1, x_t.shape, self.device)
        sigma = eta * torch.sqrt((1. - alphas_bar_prevt) / (1. - alphas_bar_t) * (1. - alphas_bar_t / alphas_bar_prevt))
        variance = sigma ** 2
        # mean
        eps_cond = self.model(x_t, t, c)
        eps_uncond = self.model(x_t, t, torch.zeros_like(c, device=self.device))
        eps = (1 + self.w) * eps_cond - self.w * eps_uncond
        predicted_x_0 = (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape, self.device) * eps) \
                        / extract(self.sqrt_alphas_bar, t, x_t.shape, self.device)
        direction_pointing_to_x_t = torch.sqrt(1. - alphas_bar_prevt - variance) * eps
        mean = torch.sqrt(alphas_bar_prevt) * predicted_x_0 + direction_pointing_to_x_t
        return mean, variance
    
    def ddim_p_sample(self,
                      x_t:torch.Tensor,
                      t:torch.Tensor,
                      t_prev:torch.Tensor,
                      eta:float,
                      c:torch.Tensor) -> torch.Tensor:
        """
        sample from p_{theta}(x_{t-1}|x_t) in DDIM
        """
        assert t.shape == (x_t.shape[0],)
        mean, variance = self.ddim_p_mean_variance(x_t, t, t_prev, eta, c)
        z = torch.randn_like(x_t)
        z[t <= 0] = 0
        return mean + torch.sqrt(variance) * z
    
    def ddim_p_sample_loop(self,
                           x_shape:tuple,
                           num_steps:int,
                           eta:float,
                           tau_mode:str,
                           c:torch.Tensor) -> torch.Tensor:
        assert tau_mode in set(['linear', 'quadratic'])
        if tau_mode == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        else:
            tseq = list((np.linspace(0, np.sqrt(self.T-1), num_steps) ** 2).astype(int))
            tseq[-1] = self.T - 1
        x_t = torch.randn(x_shape, device=self.device)
        ones = torch.ones([x_t.shape[0]], device=self.device, dtype=torch.long)
        for i in tqdm(range(num_steps), desc='DDIM Sampling', leave=False, dynamic_ncols=True):
            with torch.no_grad():
                t = ones * tseq[-1-i]
                if i != num_steps - 1:
                    t_prev = ones * tseq[-2-i]
                else:
                    t_prev = -ones
                x_t = self.ddim_p_sample(x_t, t, t_prev, eta, c)
        x_t = torch.clamp(x_t, -1, 1)
        return x_t
    
    def train_loss(self,
                   x_0:torch.Tensor,
                   c:torch.Tensor) -> torch.Tensor:
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=self.device)
        eps = torch.randn_like(x_0)
        x_t, eps = self.q_sample(x_0, t)
        eps_pred = self.model(x_t, t, c)
        loss = F.mse_loss(eps_pred, eps, reduction='mean')
        return loss