import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class MnistLoader:

    @classmethod
    def get_mnist_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            MNIST('./data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            MNIST('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=128, shuffle=True)
        
        return train_loader, val_loader

class RandomDataset(Dataset):

    def __init__(self, count, input_size, num_classes):
        self.count = count
        self.input_size = input_size
        self.num_classes = num_classes

    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        if idx > self.count:
            raise IndexError(f"IndexOutOfRange: number of items in dataset {self.count}")
        return torch.rand((self.input_size,)), torch.randint(self.num_classes, (1,))

class RandomLoader():

    @classmethod
    def get_random_loader(self, batch_size, count, input_size, num_classes):
        return DataLoader(RandomDataset(count, input_size, num_classes), batch_size=batch_size, shuffle=True)