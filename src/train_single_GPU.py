import torch
from torchvision.utils import save_image
import argparse
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_dataloader
from diffusion import GaussianDiffusion
from utils import get_betas
from UNet import UNet
from datetime import datetime
import os

def train(params:argparse.Namespace):
    dataloader = get_dataloader(params.batchsize, params.datapath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        in_ch = params.inch,
        model_ch = params.modelch,
        out_ch = params.outch,
        ch_mul = params.chmul,
        num_res_blocks = params.numres,
        cdim = params.cdim,
        num_labels = params.numlabel,
        use_conv = params.useconv,
        droprate = params.droprate,
        dtype = params.dtype
    )
    betas = get_betas(params.betamode, params.T)
    diffusion = GaussianDiffusion(
        dtp = params.dtype,
        model = net,
        betas = betas,
        w = params.w,
        dvc = device
    )
    optimizer = optim.Adam(
        diffusion.model.parameters(),
        lr = params.lr,
        weight_decay = 1e-4
    )
    for epc in tqdm(range(params.epoch), desc='Epoch', position=0, leave=False, dynamic_ncols=True):
        diffusion.model.train()
        tot_loss = 0
        for img, lab in tqdm(dataloader, desc='Processing', position=1, leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            x_0 = img.to(device)
            label = lab.to(device)
            label[np.where(np.random.rand(params.batchsize)<params.uncondrate)] = 0
            loss = diffusion.train_loss(x_0, label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        if (epc + 1) % 10 == 0:
            tqdm.write(f"Epoch {epc + 1} Loss: {tot_loss}")
        if (epc + 1) % 100 == 0:
            diffusion.model.eval()
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with torch.no_grad():
                label = torch.ones(params.numlabel, params.numgen).type(torch.long) \
                    * torch.arange(0, params.numlabel).reshape(-1, 1)
                label = label.reshape(-1, 1).squeeze()
                label = label.to(device)
                genshape = (params.numlabel * params.numgen, 3, 32, 32)
                if params.ddim:
                    samples = diffusion.ddim_p_sample_loop(genshape, params.ddimsteps, params.eta, params.taumode, label)
                else:
                    samples = diffusion.p_sample_loop(genshape, label)
                # after sample_loop, the data belongs to [-1, 1], and needs to be transformed to [0, 1]
                samples = samples / 2 + 0.5
                # samples = samples.reshape(params.numlabel, params.numgen, 3, 32, 32).contiguous()
                save_image(samples, os.path.join(params.samplepath, f'{epc + 1}_{now}.png'), nrow=params.numgen)
            mpath = os.path.join(params.modelpath, f'{epc + 1}_{now}.pt')
            torch.save({
                'epoch': epc,
                'model': diffusion.model.state_dict(),
                'optim': optimizer.state_dict(),
                'lr': params.lr,
                'loss': loss.item()
            }, mpath)
        torch.cuda.empty_cache()

def str2bool(s:str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ['true', 'yes', 't', 'y', '1']:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description='test for diffusion model')
    # train params
    parser.add_argument('--batchsize', type=int, default=256, help='batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--datapath', type=str, default='../datasets', help='data path')
    parser.add_argument('--modelpath', type=str, default='../pt', help='saved-model path')
    parser.add_argument('--samplepath', type=str, default='../samples', help='')
    parser.add_argument('--uncondrate', type=float, default=0.1, help='rate of training without condition for classifier-free guidance')
    parser.add_argument('--interval', type=int, default=100, help='epoch interval between two evaluations')
    parser.add_argument('--numgen', type=int, default=10, help='the number of samples for each label')
    parser.add_argument('--numlabel', type=int, default=10, help='num of labels')
    # diffusion params
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--T', type=int, default=1000, help='total timesteps for diffusion')
    parser.add_argument('--betamode', type=str, default='cosine', help='noise schedule mode: cosine or linear')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--ddim',type=str2bool, default=False, help='whether DDIM is used in sample, if not DDPM is used')
    parser.add_argument('--eta', type=float, default=0, help='eta for variance during DDIM sampling process')
    parser.add_argument('--taumode', type=str, default='linear', help='sub-sequence selection for reverse process in DDIM, which should be \'linear\' or \'quadratic\'')
    parser.add_argument('--ddimsteps',type=int,default=50,help='sampling steps for DDIM')
    # UNet params
    parser.add_argument('--inch',type=int,default=3,help='input channels for UNet')
    parser.add_argument('--modelch', type=int, default=64, help='model channels for UNet')
    parser.add_argument('--outch', type=int, default=3, help='output channels for UNet')
    parser.add_argument('--chmul', type=list, default=[1,2,4,8], help='architecture parameters training UNet')
    parser.add_argument('--numres', type=int, default=2, help='number of ResBlock+AttnBlock for each block in UNet')
    parser.add_argument('--cdim', type=int, default=10, help='dimension of conditional embedding')
    parser.add_argument('--useconv', type=str2bool, default=True, help='whether Conv2d is used in downsample, if not MaxPool2d is used')
    parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for ResBlock')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
