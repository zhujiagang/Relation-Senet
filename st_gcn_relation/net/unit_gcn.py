# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init
import numpy as np
from .se_module import SELayer


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        self.A = Variable(
            A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn

        # ==========================================
        self.se = SELayer(25, 2)

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()[1:]))

        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()

        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    # def forward(self, x):
    #     N, C, T, V = x.size()
    #     self.A = self.A.cuda(x.get_device())
    #     A = self.A
    #
    #     x = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C, 1)
    #     x = self.se(x).view(N, T, V, C).permute(0, 3, 1, 2).contiguous()
    #
    #     tt = np.eye(V)
    #     temp = []
    #     for i in range(V):
    #         a = A[0, i, i] * (self.mask[:, i].sum(0) - self.mask[i, i].detach())
    #         temp.append(a)
    #
    #     temp = torch.cat(temp, 0).view(1, -1).repeat(V, 1)
    #     temp = Variable(torch.from_numpy(tt).float().cuda(), requires_grad=False) * temp
    #
    #     # graph convolution
    #     for i, a in enumerate(A):
    #         if i == 0:
    #             xa = x.view(-1, V).mm(temp).view(N, C, T, V)
    #             y = self.conv_list[i](xa)
    #         else:
    #             a = a * self.mask
    #             xa = x.view(-1, V).mm(a).view(N, C, T, V)
    #             y = y + self.conv_list[i](xa)
    #
    #     # batch normalization
    #     if self.use_local_bn:
    #         y = y.permute(0, 1, 3, 2).contiguous().view(
    #             N, self.out_channels * V, T)
    #         y = self.bn(y)
    #         y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
    #     else:
    #         y = self.bn(y)
    #
    #     # nonliner
    #     y = self.relu(y)
    #     return y, self.mask

    #### N * T   14:26 - 11:28    about 3 min
    # save
    def forward(self, x):
        N, C, T, V = x.size()
        self.A = self.A.cuda(x.get_device())
        A = self.A

        ############ A1 ################
        xs = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C, 1)
        xs = self.se(xs).view(N * T, 1, V, V)
        aa = A[1].view(1, 1, V, V).repeat(N * T, 1, 1, 1)
        xxx = xs * aa

        ############ A0 ################
        tt = np.eye(V).reshape(1, 1, V, V)
        tt = np.tile(tt, (N * T, 1, 1, 1))
        temp = []
        for i in range(V):
            a = A[0, i, i] * (xs[:, :, :, i].sum(2) - xs[:, :, i, i].detach())
            temp.append(a)

        temp = torch.cat(temp, 1).view(N * T, 1, V, 1).repeat(1, 1, 1, V)
        temp = Variable(torch.from_numpy(tt).float().cuda(), requires_grad=False) * temp

        x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        # graph convolution
        for i, a in enumerate(A):
            if i == 0:
                temp = temp.view(N * T, V, V)
                xa = torch.matmul(x, temp).view(N, T, C, V).permute(0, 2, 1, 3)
                y = self.conv_list[i](xa)
            else:
                xxx = xxx.view(N * T, V, V)
                xa = torch.matmul(x, xxx).view(N, T, C, V).permute(0, 2, 1, 3)
                y = y + self.conv_list[i](xa)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)

        # nonliner
        y = self.relu(y)
        return y, self.mask


    ##### N  less than 3 min     10:03 - 7:49
    # # save

    # def forward(self, x):
    #     N, C, T, V = x.size()
    #     self.A = self.A.cuda(x.get_device())
    #     A = self.A
    #
    #     ############ A1 ################
    #     xs = x.permute(0, 3, 2, 1).contiguous()#.view(N, V, T, C)
    #     xs = self.se(xs).view(N, 1, V, V)
    #     aa = A[1].view(1, 1, V, V).repeat(N, 1, 1, 1)
    #     xxx = xs * aa
    #
    #     ############ A0 ################
    #     tt = np.eye(V).reshape(1, 1, V, V)
    #     tt = np.tile(tt, (N, 1, 1, 1))
    #     temp = []
    #     for i in range(V):
    #         a = A[0, i, i] * (xs[:, :, :, i].sum(2) - xs[:, :, i, i].detach())
    #         temp.append(a)
    #
    #     temp = torch.cat(temp, 1).view(N, 1, V, 1).repeat(1, 1, 1, V)
    #     temp = Variable(torch.from_numpy(tt).float().cuda(), requires_grad=False) * temp
    #
    #     x = x.view(N, C * T, V)
    #     # graph convolution
    #     for i, a in enumerate(A):
    #         if i == 0:
    #             temp = temp.view(N, V, V)
    #             xa = torch.matmul(x, temp).view(N, C, T, V)
    #             y = self.conv_list[i](xa)
    #         else:
    #             xxx = xxx.view(N, V, V)
    #             xa = torch.matmul(x, xxx).view(N, C, T, V)
    #             y = y + self.conv_list[i](xa)
    #
    #     # batch normalization
    #     if self.use_local_bn:
    #         y = y.permute(0, 1, 3, 2).contiguous().view(
    #             N, self.out_channels * V, T)
    #         y = self.bn(y)
    #         y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
    #     else:
    #         y = self.bn(y)
    #
    #     # nonliner
    #     y = self.relu(y)
    #     return y, self.mask