# Classifier-Free Diffusion Guidance

实现了Classifier-Free Diffusion。

使用的数据集为CIFAR-10

`python train_single_GPU.py [args]`即可开始训练。



## Parameters

> (py11) PS E:\Project\DiffusionModel\src> python train_single_GPU.py -h
> usage: train_single_GPU.py [-h] [--batchsize BATCHSIZE] [--lr LR] [--epoch EPOCH] [--datapath DATAPATH]
>                            [--modeldir MODELDIR] [--uncondrate UNCONDRATE] [--dtype DTYPE] [--T T]
>                            [--betamode BETAMODE] [--w W] [--ddim DDIM] [--eta ETA] [--taumode TAUMODE]
>                            [--ddimsteps DDIMSTEPS] [--inch INCH] [--modelch MODELCH] [--outch OUTCH] [--chmul CHMUL]
>                            [--numres NUMRES] [--cdim CDIM] [--numlable NUMLABLE] [--useconv USECONV]
>                            [--droprate DROPRATE]
>
> test for diffusion model
>
> options:
>   -h, --help            show this help message and exit
>   --batchsize BATCHSIZE
>                         batch size for training
>   --lr LR               learning rate
>   --epoch EPOCH         epoch
>   --datapath DATAPATH   data path
>   --modeldir MODELDIR   saved-model path
>   --uncondrate UNCONDRATE
>                         rate of training without condition for classifier-free guidance
>   --dtype DTYPE
>   --T T                 total timesteps for diffusion
>   --betamode BETAMODE   noise schedule mode: cosine or linear
>   --w W                 hyperparameters for classifier-free guidance strength
>   --ddim DDIM           whether DDIM is used in sample, if not DDPM is used
>   --eta ETA             eta for variance during DDIM sampling process
>   --taumode TAUMODE     sub-sequence selection for reverse process in DDIM, which should be 'linear' or 'quadratic'
>   --ddimsteps DDIMSTEPS
>                         sampling steps for DDIM
>   --inch INCH           input channels for UNet
>   --modelch MODELCH     model channels for UNet
>   --outch OUTCH         output channels for UNet
>   --chmul CHMUL         architecture parameters training UNet
>   --numres NUMRES       number of ResBlock+AttnBlock for each block in UNet
>   --cdim CDIM           dimension of conditional embedding
>   --numlable NUMLABLE   num of labels
>   --useconv USECONV     whether Conv2d is used in downsample, if not MaxPool2d is used
>   --droprate DROPRATE   dropout rate for ResBlock