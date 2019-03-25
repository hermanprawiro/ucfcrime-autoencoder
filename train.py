import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from datasets.ucfcrime import UCFCrime
from models import conv3d
from utils.metrics import AverageMeter
from utils.videotransforms import video_transforms, volume_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, optimizer, epoch, checkpoint_prefix):
    checkpoint_name = '{}.tar'.format(checkpoint_prefix)
    checkpoint_path = os.path.join('checkpoints', checkpoint_name)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)

def main():
    root_dir = "E:\\Datasets\\UCFCrime_img_best"
    learning_rate = 1e-3
    weight_decay = 0
    batch_size = 16
    start_epoch = 0
    max_epoch = 1
    base_size = 8

    checkpoint_prefix = 'conv3dae'

    train_tfs = video_transforms.Compose([
        video_transforms.Resize(224),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor()
    ])
    test_tfs = video_transforms.Compose([
        video_transforms.Resize(224),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor()
    ])

    trainset = UCFCrime(root_dir, train=True, transforms=train_tfs)
    testset = UCFCrime(root_dir, train=False, transforms=test_tfs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = conv3d.Conv3DAE(input_channels=3, base_size=base_size).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    checkpoint_path = os.path.join('checkpoints', '{}.tar'.format(checkpoint_prefix))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Checkpoint loaded, last epoch = {}'.format(checkpoint['epoch'] + 1))
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, max_epoch):
        train(trainloader, model, optimizer, criterion, None, epoch)
    visualize(testloader, model)

def train(loader, model, optimizer, criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        outs = model(inputs)
        loss = criterion(outs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)

        # global_step = (epoch * total_iter) + i + 1
        # writer.add_scalar('train/loss', losses.val, global_step)

        if i % 10 == 0:
            print('Epoch {0} [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, loss=losses)
            )

@torch.no_grad()
def visualize(loader, model):
    model.eval()

    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)

    outs = model(inputs)

    plt.subplot(2, 1, 1)
    plt.axis('off')
    plt.imshow(np.transpose(torchvision.utils.make_grid(inputs[0].permute(1, 0, 2, 3).cpu(), nrow=8, normalize=True), axes=(1, 2, 0)))
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(np.transpose(torchvision.utils.make_grid(outs[0].permute(1, 0, 2, 3).cpu(), nrow=8, normalize=True), axes=(1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    main()