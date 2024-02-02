from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(batchsize: int, dpath:str):
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
        drop_last=True
    )
    return loader
