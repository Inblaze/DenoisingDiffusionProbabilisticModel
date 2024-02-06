import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, dtp:torch.dtype, model, betas:np.ndarray, w:float, dvc:torch.device):
        super().__init__()
        self.dtp = dtp
        self.model = model.to(dvc)
        self.betas = torch.tensor(betas, dtype=dtp, device=dvc)
        self.w = w
        self.dvc = dvc
        self.T = len(betas)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)
        self.one_minus_alpha_bar = 1 - self.alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(self.one_minus_alpha_bar)
        self.recip_sqrt_alpha = 1 / torch.sqrt(self.alphas)
        self.one_minus_alpha = 1 - self.alphas
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], [1,0], 'constant', self.alpha_bar[0])
        self.one_minus_alpha_bar_prev = 1 - self.alpha_bar_prev

    @staticmethod
    def _extract(arr:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
       assert x_shape[0] == t.shape[0]
       new_shape = torch.ones_like(torch.tensor(x_shape))
       new_shape[0] = x_shape[0]
       new_shape = new_shape.tolist()
       chosen = arr[t]
       chosen = chosen.to(t.device)
       return chosen.reshape(new_shape)

    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        eps = torch.randn_like(x_0, requires_grad=False)
        x_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_0.shape) * eps \
             + self._extract(self.sqrt_alpha_bar, t, x_0.shape) * x_0
        return x_t, eps
        
    def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, c:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the mean and the variance of p_{theta}(x_{t-1}|x_{t})
        """
        assert t.shape == (x_t.shape[0],)
        # mean
        eps_cond = self.model(x_t, t, c)
        c = torch.zeros_like(c, device=self.dvc)
        eps_uncond = self.model(x_t, t, c)
        eps = (1 + self.w) * eps_cond - self.w * eps_uncond
        mean = self._extract(self.recip_sqrt_alpha, t, x_t.shape) \
                * (x_t - (self._extract(self.one_minus_alpha, t, x_t.shape) \
                / self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)) * eps)
        # variance
        variance = self._extract(self.betas, t, x_t.shape) \
                    * self._extract(self.one_minus_alpha_bar_prev, t, x_t.shape) \
                    / self._extract(self.one_minus_alpha_bar, t, x_t.shape)
        return mean, variance
    
    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        """
        sample from p_{theta}(x_{t-1}|x_t)
        """
        assert t.shape == (x_t.shape[0],)
        mean, variance = self.p_mean_variance(x_t, t, c)
        z = torch.randn_like(x_t)
        z[t <= 0] = 0
        return mean + variance * z
    
    def p_sample_loop(self, x_shape:tuple, c:torch.Tensor) -> torch.Tensor:
        x_t = torch.randn(x_shape, device=self.dvc)
        t = torch.ones([x_shape[0]], device=self.dvc) * self.T
        for _ in tqdm(range(self.T), desc='Sampling'):
            t -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, t, c)
        x_t = torch.clamp(x_t, -1, 1)
        return x_t
    
    def ddim_p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, t_prev:torch.Tensor, eta:float, c:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the mean and the variance of p_{theta}(x_{t-1}|x_{t}) in DDIM
        """
        # variance
        variance = eta * \
                torch.sqrt(self._extract(self.one_minus_alpha_bar, t_prev, x_t.shape) / self._extract(self.one_minus_alpha_bar, t, x_t.shape)) * \
                torch.sqrt(self._extract(self.one_minus_alpha_bar, t, x_t.shape) / self._extract(self.alpha_bar, t_prev, x_t.shape))
        # mean
        eps_cond = self.model(x_t, t, c)
        c = torch.zeros_like(c, device=self.dvc)
        eps_uncond = self.model(x_t, t, c)
        eps = (1 + self.w) * eps_cond - self.w * eps_uncond
        predicted_x_0 = (x_t - self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape) * eps) \
                        / self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        direction_pointing_to_x_t = torch.sqrt(self._extract(self.one_minus_alpha_bar, t_prev, x_t.shape) - variance ** 2) * eps
        mean = self._extract(self.sqrt_alpha_bar, t_prev, x_t.shape) * predicted_x_0 + direction_pointing_to_x_t
        return mean, variance
    
    def ddim_p_sample(self, x_t:torch.Tensor, t:torch.Tensor, t_prev:torch.Tensor, eta:float, c:torch.Tensor) -> torch.Tensor:
        """
        sample from p_{theta}(x_{t-1}|x_t) in DDIM
        """
        assert t.shape == (x_t.shape[0],)
        mean, variance = self.ddim_p_mean_variance(x_t, t, t_prev, eta, c)
        z = torch.randn_like(x_t)
        z[t <= 0] = 0
        return mean + variance * z
    
    def ddim_p_sample_loop(self, x_shape:tuple, num_steps:int, eta:float, tau_mode:str, c:torch.Tensor) -> torch.Tensor:
        assert tau_mode in set(['linear', 'quadratic'])
        if tau_mode == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        else:
            tseq = list(np.linspace(0, np.sqrt(self.T-1), num_steps) ** 2).astype(int)
            tseq[-1] = self.T - 1
        x_t = torch.randn(x_shape, device=self.dvc)
        ones = torch.ones([x_t.shape[0]], device=self.dvc)
        for i in tqdm(range(num_steps)):
            with torch.no_grad():
                t = ones * tseq[-1-i]
                if i != num_steps - 1:
                    t_prev = ones * tseq[-2-i]
                else:
                    t_prev = -ones
                x_t = self.ddim_p_sample(x_t, t, t_prev, eta, c)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        return x_t

    def train_loss(self, x_0:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        t = torch.randint(high=self.T, size=(x_0.shape[0],), device=self.dvc)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, c)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss