import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torchvision.models as models
import cv2
from torch.autograd import Variable
from .model_util import *
from .seg_model import DeeplabMulti

pspnet_specs = {
    'n_classes': 19,
    'block_config': [3, 4, 23, 3],
}


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=16):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class BoundaryMapping5(nn.Module):
    def __init__(self, num_input, num_output):
        super(BoundaryMapping5, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_input, num_input, 1),
                                  nn.BatchNorm2d(num_input),
                                  nn.ReLU())
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(num_input, num_input, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x, map_4):
        b_4 = x * map_4 + x
        so_output = self.conv1(b_4)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3(so_output)
        so_output = self.upsample2(so_output)

        return so_output


class BoundaryMapping2(nn.Module):
    def __init__(self, num_input, num_output):
        super(BoundaryMapping2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_input, num_input, 1),
                                  nn.BatchNorm2d(num_input),
                                  nn.ReLU())
        self.conv22 = nn.Sequential(nn.Conv2d(num_input * 2, num_input, 1),
                                    nn.BatchNorm2d(num_input),
                                    nn.ReLU())
        self.conv1x1_output = nn.Sequential(nn.Conv2d(num_input, num_input // 2, 1),
                                            nn.BatchNorm2d(num_input // 2),
                                            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x_1, map_1):
        b_1 = x_1 * map_1 + x_1
        so_output = self.conv1(b_1)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3(so_output)

        return so_output


class BoundaryMapping3(nn.Module):
    def __init__(self, num_input, num_output):
        super(BoundaryMapping3, self).__init__()
        self.conv22 = nn.Sequential(nn.Conv2d(num_input * 2, num_input, 1),
                                    nn.BatchNorm2d(num_input),
                                    nn.ReLU())
        self.conv1x1 = nn.Sequential(nn.Conv2d(num_input, num_input, 1),
                                     nn.BatchNorm2d(num_input),
                                     nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_input // 2, num_input // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_input // 4, num_output, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x_2, map_2):
        b_2 = x_2 * map_2 + x_2
        so_output = self.conv1(b_2)
        so_output = self.upsample2(so_output)
        so_output = self.conv2(so_output)
        so_output = self.upsample2(so_output)
        so_output = self.conv3(so_output)
        so_output = self.upsample2(so_output)
        return so_output


class GenMap(nn.Module):
    def __init__(self, num_input):
        super(GenMap, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input // 2, num_input, kernel_size=3, padding=1, bias=False)
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_input, num_input // 16, bias=True)

        self.fc2 = nn.Linear(num_input // 16, num_input, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        x = self.conv3_1(x)
        x = self.pooling(x)
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = fc_out_2.view(a, b, 1, 1)

        return output_tensor


class FeatureMapping(nn.Module):
    def __init__(self, num_input):
        super(FeatureMapping, self).__init__()
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input // 2, num_input // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input // 2, num_input, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv3_1(x)

        return x


class SNR(nn.Module):

    def __init__(self, input_channels, reduction_ratio=16):
        super(SNR, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio
        self.InstanceNorm = nn.InstanceNorm2d(self.input_channels, affine=True)
        self.SE = ChannelSELayer(self.input_channels, self.reduction_ratio)

    def forward(self, f):
        ff = self.InstanceNorm(f)
        r = f - ff
        r_plus = self.SE(r)
        ff_plus = ff + r_plus
        return ff_plus


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']

        Seg_Model = DeeplabMulti(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4
        self.classifer3 = Seg_Model.layer5
        self.classifer4 = Seg_Model.layer6

        self.IN_1 = SNR(256)
        self.IN_2 = SNR(512)
        self.IN_3 = SNR(1024)
        self.IN_4 = SNR(2048)

        self.Mapping_1 = FeatureMapping(256)
        self.Mapping_2 = FeatureMapping(512)
        self.Mapping_3 = FeatureMapping(1024)
        self.Mapping_4 = FeatureMapping(2048)

        self.fb_1 = BoundaryMapping2(256, 1)
        self.fb_2 = BoundaryMapping3(512, 1)
        self.fb_3 = BoundaryMapping3(1024, 1)
        self.fb_4 = BoundaryMapping5(2048, 1)

        self.score_final = nn.Conv2d(4, 1, 1)
        self.genmap_1 = GenMap(256)
        self.genmap_2 = GenMap(512)
        self.genmap_3 = GenMap(1024)
        self.genmap_4 = GenMap(2048)

    def forward(self, x, label=None):
        low = self.layer0(x)
        x_1 = self.layer1(low)
        x_1 = self.IN_1(x_1)
        map_1 = self.genmap_1(x_1)
        boundary_1 = self.fb_1(x_1, map_1)
        x_1 = x_1 * map_1.detach() + x_1
        x_2 = self.layer2(x_1)
        x_2 = self.IN_2(x_2)
        map_2 = self.genmap_2(x_2)
        boundary_2 = self.fb_2(x_2, map_2)
        x_2 = x_2 * map_2.detach() + x_2
        x_3 = self.layer3(x_2)
        x_3 = self.IN_3(x_3)
        map_3 = self.genmap_3(x_3)
        boundary_3 = self.fb_3(x_3, map_3)
        x_3 = x_3 * map_3.detach() + x_3
        x_4 = self.layer4(x_3)
        x_4 = self.IN_4(x_4)
        map_4 = self.genmap_4(x_4)
        boundary_4 = self.fb_4(x_4, map_4)
        x_4 = x_4 * map_4.detach() + x_4

        x3 = self.classifer3(x_3)
        x4 = self.classifer4(x_4)

        fusecat = torch.cat((boundary_4, boundary_3, boundary_2, boundary_1), dim=1)
        fuse = self.score_final(fusecat)

        results = [boundary_4, boundary_3, boundary_2, boundary_1, fuse]
        results = [torch.sigmoid(r) for r in results]

        if label is not None:

            label_class_1 = (label == 0).float()
            label_class_1 = torch.unsqueeze(label_class_1, dim=0)
            label_class_1 = F.interpolate(label_class_1, scale_factor=0.125, mode='nearest')
            label_class_sum_1 = torch.sum(label_class_1, dim=(2, 3))

            x_4_class_1_mean = x_4 * label_class_1
            x_4_class_1_mean = torch.sum(x_4_class_1_mean, dim=(2, 3), keepdim=True) / (label_class_sum_1 + 1e-10)
            x_4_minus_1 = torch.squeeze(x_4 - x_4_class_1_mean, dim=2)
            x_4_minus_1 = torch.squeeze(x_4_minus_1, dim=2)
            deviation_1 = torch.sum((x_4_minus_1)**2 * label_class_1) / (label_class_sum_1 + 1e-10)
            class3_contrastive = torch.unsqueeze(deviation_1, dim=1)

            x_4_class_1 = x_4 * label_class_1
            x_4_class_1 = torch.sum(x_4_class_1, dim=(2, 3)) / (label_class_sum_1 + 1e-10)
            class4_contrastive = torch.unsqueeze(x_4_class_1, dim=1)

            for i in range(18):
                label_class = (label == i + 1).float()
                label_class = torch.unsqueeze(label_class, dim=0)
                label_class = F.interpolate(label_class, scale_factor=0.125, mode='nearest')
                label_class_sum = torch.sum(label_class, dim=(2, 3))

                x_4_class_mean = x_4 * label_class
                x_4_class_mean = torch.sum(x_4_class_mean, dim=(2, 3), keepdim=True) / (label_class_sum + 1e-10)
                x_4_minus = torch.squeeze(x_4 - x_4_class_mean, dim=2)
                x_4_minus = torch.squeeze(x_4_minus, dim=2)
                deviation = torch.sum((x_4_minus) ** 2 * label_class) / (label_class_sum + 1e-10)
                x_4_class = torch.unsqueeze(deviation, dim=1)
                class3_contrastive = torch.cat((class3_contrastive, x_4_class), dim=1)

                x_4_class = x_4 * label_class
                x_4_class = torch.sum(x_4_class, dim=(2, 3)) / (label_class_sum + 1e-10)
                x_4_class = torch.unsqueeze(x_4_class, dim=1)
                class4_contrastive = torch.cat((class4_contrastive, x_4_class), dim=1)

            return class3_contrastive, class4_contrastive, x3, x4, results, torch.sigmoid(fuse), torch.sigmoid(boundary_4), torch.sigmoid(
                boundary_3), torch.sigmoid(boundary_2), torch.sigmoid(boundary_1)

        return x4

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.classifer3.parameters())
        b.append(self.classifer4.parameters())
        b.append(self.genmap_1.parameters())
        b.append(self.genmap_2.parameters())
        b.append(self.genmap_3.parameters())
        b.append(self.genmap_4.parameters())
        b.append(self.IN_1.parameters())
        b.append(self.IN_2.parameters())
        b.append(self.IN_3.parameters())
        b.append(self.IN_4.parameters())
        b.append(self.fb_3.parameters())
        b.append(self.fb_2.parameters())
        b.append(self.fb_1.parameters())
        b.append(self.fb_4.parameters())
        b.append(self.score_final.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


