from datetime import datetime
import os
import torch
from torchvision.utils import save_image
import argparse

from diffusion import Diffusion
from UNet import UNet
from utils import get_betas, str2bool

def eval(params:argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(T=params.T,
               num_labels=params.numlabel,
               ch=params.modelch,
               ch_mult=params.chmul,
               num_res_blocks=params.numres,
               dropout=params.dropout)
    betas = get_betas(params.betamode, params.T)
    diffusion = Diffusion(
        model = net,
        betas = betas,
        w = params.w,
        device = device
    )
    mpath = os.path.join(params.modeldir, params.modelfilename)
    diffusion.model.load_state_dict(torch.load(mpath)['model'])
    diffusion.model.eval()
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    with torch.no_grad():
        label = torch.ones(params.numlabel, params.numgen).type(torch.long) \
            * torch.arange(1, params.numlabel+1).reshape(-1, 1)
        label = label.reshape(-1, 1).squeeze()
        label = label.to(device)
        genshape = (params.numlabel * params.numgen, 3, 32, 32)
        if params.ddim:
            samples = diffusion.ddim_p_sample_loop(genshape, params.ddimsteps, params.eta, params.taumode, label)
        else:
            samples = diffusion.p_sample_loop(genshape, label)
        # after sample_loop, the data belongs to [-1, 1], and needs to be transformed to [0, 1]
        samples = samples * 0.5 + 0.5
        # samples = samples.reshape(params.numlabel, params.numgen, 3, 32, 32).contiguous()
        sample_mode = 'ddim' if params.ddim else 'ddpm'
        save_image(samples, os.path.join(params.sampledir, f'eval_{sample_mode}_{now}.png'), nrow=params.numgen)

def main():
    parser = argparse.ArgumentParser(description='diffusion model evaluation')

    parser.add_argument('--modeldir', type=str, default='../pt', help='saved-model directory')
    parser.add_argument('--modelfilename', type=str, required=True, help='saved-model filename')
    parser.add_argument('--sampledir', type=str, default='../samples', help='sample directory')
    parser.add_argument('--numgen', type=int, default=10, help='the number of samples for each label')
    parser.add_argument('--numlabel', type=int, default=10, help='num of labels')

    parser.add_argument('--T', type=int, default=1000, help='total timesteps for diffusion')
    parser.add_argument('--betamode', type=str, default='linear', help='noise schedule mode: cosine or linear')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--ddim',type=str2bool, default=False, help='whether DDIM is used in sampling. If not, DDPM is used.')
    parser.add_argument('--eta', type=float, default=0, help='eta for variance during DDIM sampling process')
    parser.add_argument('--taumode', type=str, default='linear', help='sub-sequence selection for reverse process in DDIM, which should be \'linear\' or \'quadratic\'')
    parser.add_argument('--ddimsteps',type=int,default=50,help='sampling steps for DDIM')

    parser.add_argument('--modelch', type=int, default=64, help='model channels for UNet')
    parser.add_argument('--chmul', type=list, default=[1,2,2,2], help='architecture parameters training UNet')
    parser.add_argument('--numres', type=int, default=2, help='number of ResBlock+AttnBlock for each block in UNet')
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate for ResBlock')

    args = parser.parse_args()
    eval(args)

if __name__=='__main__':
    main()