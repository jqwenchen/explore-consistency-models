from tqdm import tqdm
import math
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import csv
import argparse
from torch.autograd import Variable
from apex import amp
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data.distributed import DistributedSampler
from consistency_models import ConsistencyModel, kerras_boundaries
from dataset_img_9 import get_imagenet_dataloader
import os

def imagenet9_dl():
    input_size= 224
    batch_size = 16
    tf = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.RandomCrop([input_size, input_size], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = batch_size
    train_dir = './data/imagenet-1k/train'
    test_dir = './data/imagenet-1k/val'

    trainset = get_imagenet_dataloader(train_dir, batch_size=batch_size,
                                       transform=tf, train=True,
                                       val_data='ImageNet',)

    # train_sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=1,shuffle=True)
    return trainloader

def train(
    args,
    n_epoch: int = 100,
    device="cuda:0",
    dataloader=imagenet9_dl(),
    n_channels=1,
    name="mnist",
):
    model = ConsistencyModel(n_channels, D=256)
    '''
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True).to(device)
    '''
    model = model.to(device)
    # model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(n_channels, D=256)
    '''
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True).to(device)
    '''
    ema_model = ema_model.to(device)
    cudnn.benchmark = True
    # ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")

        model.eval()
        with torch.no_grad():
            # Sample 5 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_5step_{epoch}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_2step_{epoch}.png")

            # save model
            torch.save(model.state_dict(), f"./ct_{name}.pth")

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-9 Training')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')

    parser.add_argument('--input_size', default=224, type=int,
                        help='the size of input image')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='total epochs to run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--severity', type=int, default=3)

    args = parser.parse_args()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    # torch.distributed.init_process_group(backend='nccl')  # 初始化"gloo"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "11111"

    train(args, dataloader=imagenet9_dl(), n_channels=3, name="imagenet-9")


if __name__ == '__main__':
    main()