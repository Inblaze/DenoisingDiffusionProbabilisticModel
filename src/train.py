from datetime import datetime
import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np

from diffusion import Diffusion
from utils import get_named_beta_schedule, str2bool, get_model
from utils import get_cifar_dataloader
# from utils import get_agfw_dataloader

def train(params:argparse.Namespace):
    dataloader = get_cifar_dataloader(params.batchsize, params.datadir)
    # dataloader = get_agfw_dataloader(params.batchsize, params.datadir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(T=params.T,
                    num_labels=params.numlabel,
                    ch=params.modelch,
                    img_size=params.imgsz,
                    num_res_blocks=params.numres,
                    dropout=params.dropout)
    betas = get_named_beta_schedule(params.betamode, params.T)
    diffusion = Diffusion(
        model = net,
        betas = betas,
        w = params.w,
        device = device
    )
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=params.lr)
    logger = SummaryWriter(params.logdir)
    for epc in tqdm(range(params.epoch), desc='Epoch', position=0, leave=False, dynamic_ncols=True):
        diffusion.model.train()
        tot_loss = 0
        for img, lab in tqdm(dataloader, desc='Processing', position=1, leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            x_0 = img.to(device)
            label = (lab + 1).to(device) # lab in [0, numlabel-1] so (lab+1) in [1, numlabel]
            if np.random.rand() < params.uncondrate:
                label = torch.zeros_like(label).to(device) # lab = 0 means unconditional
            loss = diffusion.train_loss(x_0, label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        logger.add_scalar('Total Loss', tot_loss, epc+1)
        tqdm.write(f"Epoch {epc + 1} Loss: {tot_loss}")
        if (epc + 1) % params.intvleval == 0:
            diffusion.model.eval()
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with torch.no_grad():
                label = torch.ones(params.numlabel, params.numgen).type(torch.long) \
                    * torch.arange(1, params.numlabel+1).reshape(-1, 1)
                label = label.reshape(-1, 1).squeeze()
                label = label.to(device)
                genshape = (params.numlabel * params.numgen, 3, params.imgsz, params.imgsz)
                if params.ddim:
                    samples = diffusion.ddim_p_sample_loop(genshape, params.ddimsteps, params.eta, params.taumode, label)
                else:
                    samples = diffusion.p_sample_loop(genshape, label)
                # after sample_loop, the data belongs to [-1, 1], and needs to be transformed to [0, 1]
                samples = samples * 0.5 + 0.5
                # samples = samples.reshape(params.numlabel, params.numgen, 3, 32, 32).contiguous()
                save_image(samples, os.path.join(params.sampledir, f'{epc + 1}_{now}.png'), nrow=params.numgen)
        if (epc + 1) % params.intvlsave == 0:
            mpath = os.path.join(params.modeldir, f'{epc + 1}_{now}.pt')
            torch.save({
                'epoch': epc,
                'model': diffusion.model.state_dict(),
                'optim': optimizer.state_dict(),
                'lr': params.lr,
                'tot_loss': tot_loss
            }, mpath)
        torch.cuda.empty_cache()
    logger.close()

def main():
    parser = argparse.ArgumentParser(description='diffusion model training')
    # train params
    parser.add_argument('--batchsize', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--datadir', type=str, default='../datasets', help='data directory')
    parser.add_argument('--modeldir', type=str, default='../pt', help='saved-model directory')
    parser.add_argument('--sampledir', type=str, default='../samples', help='sample directory')
    parser.add_argument('--uncondrate', type=float, default=0.1, help='rate of training without condition for classifier-free guidance')
    parser.add_argument('--numgen', type=int, default=10, help='the number of samples for each label')
    parser.add_argument('--numlabel', type=int, default=10, help='num of labels')
    parser.add_argument('--logdir', type=str, default='../logs', help='log directory')
    parser.add_argument('--intvleval', type=int, default=10, help='interval of evaluation')
    parser.add_argument('--intvlsave', type=int, default=100, help='interval of saving model')
    # diffusion params
    parser.add_argument('--T', type=int, default=1000, help='total timesteps for diffusion')
    parser.add_argument('--betamode', type=str, default='linear', help='noise schedule mode: cosine or linear')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--ddim',type=str2bool, default=False, help='whether DDIM is used in sampling. If not, DDPM is used')
    parser.add_argument('--eta', type=float, default=0, help='eta for variance during DDIM sampling process')
    parser.add_argument('--taumode', type=str, default='linear', help='sub-sequence selection for reverse process in DDIM, which should be \'linear\' or \'quadratic\'')
    parser.add_argument('--ddimsteps',type=int,default=50,help='sampling steps for DDIM')
    # UNet params
    # parser.add_argument('--inch',type=int,default=3,help='input channels for UNet')
    # parser.add_argument('--outch', type=int, default=3, help='output channels for UNet')
    parser.add_argument('--modelch', type=int, default=64, help='model channels for UNet')
    parser.add_argument('--imgsz', type=int, default=32, help='image size')
    # parser.add_argument('--chmul', type=list, default=[1,2,2,2], help='architecture parameters training UNet')
    parser.add_argument('--numres', type=int, default=2, help='number of ResBlock+AttnBlock for each block in UNet')
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate for ResBlock')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
