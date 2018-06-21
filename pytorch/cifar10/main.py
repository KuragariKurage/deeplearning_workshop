import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
from network import CifarConvNet


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size for training')
    parser.add_argument('--test-batch-size', default=1000, type=int,
                        help='batch size for validation')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='SGD momentum')
    parser.add_argument('--no-cuda', default=False,
                        action='store_true', help='disable CUDA training')
    parser.add_argument('--seed', default=1, type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', default=10, type=int, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    return args


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def valid(args, model, device, valid_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output,
                                        target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


def main():

    args = arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': 1} if use_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False,
                         transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    model = CifarConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        valid(args, model, device, valid_loader)


if __name__ == '__main__':
    main()
