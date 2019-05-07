import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class Net(nn.Module):
    def __init__(self, batch_size, genome):
        super(Net, self).__init__()
        self.batch_size = batch_size

        moduleList = []
        for i in range(len(genome)-1):
            moduleList.append(nn.Linear(genome[i], genome[i+1]))

        self.features = nn.ModuleList(moduleList)

    def forward(self, x):
        for l in enumerate(self.features):
            x = l[1](x)
        return F.log_softmax(x, dim=1)


def main(genome):
    use_gpu = False
    if use_gpu and torch.cuda.is_available():
        name = "Eric"
        device_num = 0 if ord(name[0]) % 2 == 0 else 1
        torch.cuda.set_device(device_num)
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    batch_size = 5
    #g = [784, 15, 12, 10]
    #g = [784, 10]
    net = Net(batch_size, genome).to(device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    learning_rate = 0.0003
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    net.train()
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            data, target = data.to(device), target.to(device)

            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % 1000 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.item()))

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            data, target = data.to(device), target.to(device)
            output = net(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
