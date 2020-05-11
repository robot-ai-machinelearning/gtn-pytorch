import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner(nn.Module):

    def __init__(self, config):
        super(Learner, self).__init__()

        self.conv1 = nn.Conv2d(1, config.learner_conv1_filters, 3)
        self.bn_conv1 = nn.BatchNorm2d(config.learner_conv1_filters)

        self.conv2 = nn.Conv2d(config.learner_conv1_filters, config.learner_conv2_filters, 3)
        self.bn_conv2 = nn.BatchNorm2d(config.learner_conv2_filters)

        conv1_out_size = (config.learner_input_size - 3 + 1) // 2
        conv2_out_size = (conv1_out_size - 3 + 1) // 2

        self.fc = nn.Linear(config.learner_fc_filters * conv2_out_size * conv2_out_size, config.learner_classes)
        nn.init.kaiming_normal_(self.fc.weight, config.learner_leaky_relu_alpha)
        self.bn_fc = nn.BatchNorm1d(config.learner_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.config.learner_leaky_relu_alpha)
        x = self.bn_conv1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x, self.config.learner_leaky_relu_alpha)
        x = self.bn_conv2(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)

        # Log probability
        output = F.log_softmax(x, dim=1)
        return output