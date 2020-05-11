import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        # Fully Connected layer initialisation
        self.fc1 = nn.Linear(config.generator_noise_size + config.generator_classes, config.generator_fc1_filters)
        nn.init.kaiming_normal_(self.fc1.weight, config.generator_leaky_relu_alpha)
        self.bn_fc1 = nn.BatchNorm1d(config.generator_fc1_filters)
        fc2_out_filters = config.generator_fc2_filters * config.generator_output_size//4 * config.generator_output_size//4
        self.fc2 = nn.Linear(config.generator_fc1_filters, fc2_out_filters)
        nn.init.kaiming_normal_(self.fc2.weight, config.generator_leaky_relu_alpha)
        self.bn_fc2 = nn.BatchNorm1d(config.generator_fc2_filters)

        # Conv layer initialisation
        self.conv1 = nn.Conv2d(fc2_out_filters, config.generator_conv1_filters, 3)
        self.bn_conv1 = nn.BatchNorm2d(config.generator_conv1_filters)
        self.conv2 = nn.Conv2d(config.generator_conv1_filters, config.generator_conv2_filters, 3)
        self.bn_conv2 = nn.BatchNorm2d(config.generator_conv2_filters)

        self.tanh = nn.Tanh()
    
    def forward(self, x, target):
        x = torch.cat([x, target], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, self.config.generator_leaky_relu_alpha)
        x = self.bn_fc1(x)
        
        x = self.fc2(x)
        x = F.leaky_relu(x, self.config.generator_leaky_relu_alpha)
        x = self.bn_fc2(x)

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, self.config.generator_leaky_relu_alpha)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x, self.config.generator_leaky_relu_alpha)

        x = self.tanh(x)
        return x, target