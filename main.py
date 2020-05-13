import random
import torch
import numpy as np

from gtn.config.config import LearnerConfig, GeneratorConfig
from gtn.models import Learner, Generator
from gtn.datasets.datasets import MnistLoader, RandomLoader

def num_to_one_hot_encoding(batch_targets):

    targets = []
    for i in batch_targets:
        one_hot_target = np.zeros(10)
        one_hot_target[batch_targets[i].item() % 10] = 1
        targets.append(one_hot_target)

    return torch.tensor(targets)

random.seed(1)

train_loader, val_loader = MnistLoader.get_mnist_loaders()
random_loader = RandomLoader.get_random_loader(128, 128, GeneratorConfig.generator_noise_size, GeneratorConfig.generator_classes)

generator = Generator(GeneratorConfig)
learner = Learner(LearnerConfig(random.randint(32, 128), random.randint(64, 256), random.randint(64, 256), 10))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
generator.train()
generator.to(device)
learner.train()
learner.to(device)

# 100 epochs for the outer loop
zeros = np.zeros(10)
for i in range(100):
    # Get input to generator
    for data, targets in random_loader:
        data.to(device)
        one_hot_target = zeros.copy()
        targets = num_to_one_hot_encoding(targets)