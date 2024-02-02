from datetime import datetime
from UNet import *
from DiffusionModule import *
import torch.nn.functional as F


def get_loss(noise_predictor, x, t, diffusion_module, T=300):
    noisy_x, noise = diffusion_module(x, t, mode='cosine', tot_steps=T)
    pred_noise = noise_predictor(noisy_x, t)
    return F.mse_loss(pred_noise, noise)


def denoise(x_shape, lastT, noise_predictor, beta_scheduler, device):
    with torch.no_grad():
        ba_sz = x_shape[0]
        x_T = torch.randn(x_shape).to(device)
        betas = beta_scheduler(lastT)
        alphas = (1 - betas).to(device)
        sqrt_alphas = torch.sqrt(alphas).to(device)
        alpha_bar = torch.cumprod(alphas, dim=0).to(device)
        one_minus_alpha_bar = (1 - alpha_bar).to(device)
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar).to(device)
        x_t = x_T
        for t in range(lastT - 1, -1, -1):
            if t > 0:
                z = torch.randn(x_shape).to(device)
            else:
                z = torch.zeros(x_shape).to(device)
            epsilon = noise_predictor(x_t, torch.full((ba_sz,), t).long())
            variance = torch.Tensor(betas[t] * one_minus_alpha_bar[t - 1] / one_minus_alpha_bar[t]).to(device)
            x_t_minus_one = (1 / sqrt_alphas[t]) * (x_t - (1 - alphas[t]) / sqrt_one_minus_alpha_bar[t] * epsilon) \
                            + variance * z
            x_t = x_t_minus_one
        return x_t

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dpath = '../datasets/Flower102'
    batchsize = 128
    dloader = get_dataloader(dpath, img_size=64, batchsize=batchsize)
    num_batch = len(dloader)
    log_path = '../logs/Flower102'
    logger = SummaryWriter(log_path)
    model = SimpleUnet(3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100  # Try more!
    my_diffusion_module = DiffusionModule()
    T = 300
    for epoch in range(1, epochs + 1):
        tot_loss = 0
        for step, batch in enumerate(dloader):
            t = torch.randint(0, T, (batchsize,), device=device).long()
            loss = get_loss(model, batch[0], t, my_diffusion_module, T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            if epoch % 5 == 0 and step == num_batch - 1:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            # print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        logger.add_scalar("Mean Loss", tot_loss / num_batch, epoch)

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    torch.save(model, "../saves/{}.pth".format(now))
    model.eval()
    gen = denoise([8, 3, 64, 64], lastT=T, noise_predictor=model,
                  beta_scheduler=my_diffusion_module.cosine_beta_scheduler, device=device)
    gen = ((gen + 1) / 2) * 255
    tensor2PIL = torchvision.transforms.ToPILImage(mode="RGB")
    for i in range(8):
        tensor2PIL(gen[i]).save("../out/{}_{}.jpg".format(now, i))
