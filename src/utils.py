import numpy as np
import math


def get_betas(schedule_mode='linear', tot_timesteps=1000, max_beta=0.999) -> np.ndarray:
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
        return betas
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_mode}")
