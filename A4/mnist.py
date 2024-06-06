from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from src.matmul import svd


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None, bias=False):
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Initialize weight as a parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = svd(x, self.weight.t(), rank_A=None, rank_B=self.rank)
        if self.bias is not None:
            output += self.bias
        return output


def determine_rank(size, compression_ratio):
    return round(compression_ratio * size)

class Net(nn.Module):
    def __init__(self, ratio):
        super(Net, self).__init__()
        
        use_svd = True
        if use_svd:
            print("RANK: ", determine_rank(512, ratio))
            print("RANK: ", determine_rank(128, ratio))
            print("RANK: ", determine_rank(10, ratio))
            self.fc1 = SVDLinear(28 * 28, 512, determine_rank(512, ratio))
            self.fc2 = SVDLinear(512, 128, determine_rank(128, ratio))
            self.fc3 = SVDLinear(128, 10, determine_rank(10, ratio))
        else:
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    average_loss = total_loss / total_batches
    print(f'Train Epoch: {epoch} \tAverage Loss: {average_loss:.6f}')


def test(model, device, test_loader):
    model.eval()
    synchronize = False
    if device.type == "cuda":
        synchronize = True
    test_loss = 0
    correct = 0
    total_time = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get timing info
            if synchronize:
                torch.cuda.synchronize() 
            start_time = time.time()
            output = model(data)
            if synchronize:
                torch.cuda.synchronize() 
            total_time += time.time() - start_time
            num_samples += data.size(0)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    average_time_per_prediction = (total_time / num_samples) * 1000 # ms
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f"Average time per predicion: {average_time_per_prediction}")
    return accuracy, average_time_per_prediction


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--ratio', type=float, default=1.0, metavar='R',
                        help='compression ratio')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(args.ratio).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_fc.pt")



if __name__ == '__main__':
    main()