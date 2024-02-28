import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


class Net(nn.Module):
    def __init__(self, hidden_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs to train (default: 3)')
parser.add_argument('--batch_size_train', type=int, default=256,
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch_size_test', type=int, default=10000,
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--nni', type=int, default=0)
args = parser.parse_args()

if args.nni == 1:
    import nni
    from nni.utils import merge_parameter
    nni_params = nni.get_next_parameter()
    args = merge_parameter(args, nni_params)

torch.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=args.batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=args.batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

network = Net(args.hidden_dim)
optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)

def train():
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct/len(test_loader.dataset)

if __name__ == '__main__':
    best_acc = 0
    for i in range(args.n_epochs):
        train()
        test_loss, acc = test()
        if acc > best_acc:
            best_acc = acc
        if args.nni == 1:
            nni.report_intermediate_result({'default': acc, 'loss': test_loss})
    if args.nni == 1:
        nni.report_final_result({'default': best_acc})
