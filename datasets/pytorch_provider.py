import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_loaders(batch_size):
    trainset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=True,
        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=4,
        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=False,
        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=4,
        shuffle=False, num_workers=2)

    return train_loader, test_loader
