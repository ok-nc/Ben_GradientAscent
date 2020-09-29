"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""
# Own modules

# Built in
import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square, abs
from okFunc import scale_grad
from ben_division import *

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        # Set up whether this uses a Lorentzian oscillator, this is a boolean value
        self.use_lorentz = flags.use_lorentz
        self.use_conv = flags.use_conv
        self.flags = flags
        self.delta = self.flags.delta

        if flags.use_lorentz:

            # Create the constant for mapping the frequency w
            w_numpy = np.arange(flags.freq_low, flags.freq_high,
                                (flags.freq_high - flags.freq_low) / self.flags.num_spec_points)

            self.epsilon_inf = torch.tensor([5 + 0j], dtype=torch.cfloat)

            # Create the tensor from numpy array
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.w = torch.tensor(w_numpy).cuda()
                self.epsilon_inf = self.epsilon_inf.cuda()
            else:
                self.w = torch.tensor(w_numpy)

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            # torch.nn.init.uniform_(self.linears[ind].weight, a=1, b=2)

            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        layer_size = flags.linear[-1]

        # # Parallel network test
        # layer_size = 10
        # self.input_layer = nn.Linear(4,layer_size)
        # self.bn1 = nn.BatchNorm1d(layer_size)
        # self.g1 = nn.Linear(layer_size,layer_size)
        # self.g2 = nn.Linear(layer_size, layer_size)
        # self.g3 = nn.Linear(layer_size, layer_size)
        # self.bn_g1 = nn.BatchNorm1d(layer_size)
        # self.bn_g2 = nn.BatchNorm1d(layer_size)
        # self.bn_g3 = nn.BatchNorm1d(layer_size)

        self.lin_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=True)
        self.lin_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=True)
        self.lin_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=True)

        #self.bn_w0 = nn.BatchNorm1d(self.flags.num_lorentz_osc)
        #self.bn_wp = nn.BatchNorm1d(self.flags.num_lorentz_osc)
        #self.bn_g = nn.BatchNorm1d(self.flags.num_lorentz_osc)

        # self.lin_w0 = nn.Linear(self.flags.linear[-1], self.flags.num_lorentz_osc, bias=False)
        # self.lin_wp = nn.Linear(self.flags.linear[-1], self.flags.num_lorentz_osc, bias=False)
        # self.lin_g = nn.Linear(self.flags.linear[-1], self.flags.num_lorentz_osc, bias=False)
        # torch.nn.init.uniform_(self.lin_w0.weight, a=self.flags.freq_low, b=self.flags.freq_high)
        # torch.nn.init.uniform_(self.lin_w0.weight, a=2, b=4)
        # torch.nn.init.uniform_(self.lin_wp.weight, a=2, b=4)
        #torch.nn.init.uniform_(self.lin_g.weight, a=0.0, b=0.1)
        # nn.init.xavier_uniform_(self.lin_w0.weight)
        # nn.init.xavier_uniform_(self.lin_wp.weight)

        # self.divNN = div_NN()
        # for param in self.divNN.parameters():
        #     param.requires_grad = False

        if flags.use_conv:
            # Conv Layer definitions here
            self.convs = nn.ModuleList([])
            in_channel = 1                                                  # Initialize the in_channel number
            for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                         flags.conv_kernel_size,
                                                                         flags.conv_stride)):
                if stride == 2:     # We want to double the number
                    pad = int(kernel_size/2 - 1)
                elif stride == 1:   # We want to keep the number unchanged
                    pad = int((kernel_size - 1)/2)
                else:
                    Exception("Now only support stride = 1 or 2, contact Ben")

                self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                    stride=stride, padding=pad)) # To make sure L_out double each time
                in_channel = out_channel # Update the out_channel

            self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G
        # initialize the out
        # Monitor the gradient list
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind < len(self.linears) - 0:
                # out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
                out = F.relu(fc(out))                                   # ReLU + BN + Linear
            else:
                out = fc(out)
                # out = bn(fc(out))

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.leaky_relu(self.lin_w0(F.relu(out)))
            wp = F.leaky_relu(self.lin_wp(F.relu(out)))
            g = F.leaky_relu(self.lin_g(F.relu(out)))

            #w0 = F.relu(self.lin_w0(F.relu(out)))
            #wp = F.relu(self.lin_wp(F.relu(out)))
            #g = F.relu(self.lin_g(F.relu(out)))

            #w0 = pow(self.lin_w0(F.relu(out)), 2)
            #wp = pow(self.lin_wp(F.relu(out)), 2)
            #g = pow(self.lin_g(F.relu(out)), 2)

            w0_out = w0
            wp_out = wp
            g_out = g

            w0 = w0.unsqueeze(2) * 1
            wp = wp.unsqueeze(2) * 1
            g = g.unsqueeze(2) * 0.1

            w0 = w0.expand(out.size(0), self.flags.num_lorentz_osc, self.flags.num_spec_points)
            wp = wp.expand_as(w0)
            g = g.expand_as(w0)
            w_expand = self.w.expand_as(g)

            # Define dielectric function (real and imaginary parts separately)
            num1 = mul(square(wp), add(square(w0), -square(w_expand)))
            num2 = mul(square(wp), mul(w_expand, g))
            denom = add(square(add(square(w0), -square(w_expand))), mul(square(w_expand), square(g)))
            e1 = div(num1, denom)
            e2 = div(num2, denom)

            # self.e2 = e2.data.cpu().numpy()                 # This is for plotting the imaginary part
            # self.e1 = e1.data.cpu().numpy()                 # This is for plotting the imaginary part

            e1 = torch.sum(e1, 1).type(torch.cfloat)
            e2 = torch.sum(e2, 1).type(torch.cfloat)
            eps_inf = self.epsilon_inf.unsqueeze(1).expand_as(e1)
            e1 += eps_inf
            j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2, j))
            n = sqrt(eps)
            d, _ = torch.max(G[:, self.flags.num_lorentz_osc:], dim=1)
            d = d.unsqueeze(1).expand_as(n)
            # d = G[:,1].unsqueeze(1).expand_as(n)
            if self.flags.normalize_input:
                d = d * (self.flags.geoboundary[-1] - self.flags.geoboundary[-2]) * 0.5 + (
                            self.flags.geoboundary[-1] + self.flags.geoboundary[-2]) * 0.5
            alpha = torch.exp(-0.0005 * 4 * math.pi * mul(d, n.imag))

            # R = div(square(n.real - 1) + square(n.imag), square(n.real + 1) + square(n.imag))
            # R = div(square((n-1).abs()),square((n+1).abs())).float()
            # T_coeff = ones - R
            T = mul(div(4 * n.real, add(square(n.real + 1), square(n.imag))), alpha).float()

            return T, w0_out, wp_out, g_out

        # The normal mode to train without Lorentz
        if self.use_conv:
            out = out.unsqueeze(1)                                          # Add 1 dimension to get N,L_in, H
            # For the conv part
            for ind, conv in enumerate(self.convs):
                out = conv(out)

            # Final touch, because the input is normalized to [-1,1]
            # S = tanh(out.squeeze())
            out = out.squeeze()
        return out, out, out, out

# class div_NN(nn.Module):
#     def __init__(self):
#         super(div_NN, self).__init__()
#
#         linear_layers = [2, 1000, 1]
#
#         self.linears = nn.ModuleList([])
#         self.bn_linears = nn.ModuleList([])
#         for ind, fc_num in enumerate(linear_layers[0:-1]):  # Excluding the last one as we need intervals
#             self.linears.append(nn.Linear(fc_num, linear_layers[ind + 1], bias=True))
#             self.bn_linears.append(nn.BatchNorm1d(linear_layers[ind + 1], track_running_stats=True, affine=True))
#
#
#     def forward(self, input):
#
#         out = input
#
#         for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
#             #print(out.size())
#             if ind < len(self.linears) - 0:
#                 out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
#             else:
#                 out = bn(fc(out))
#
#         return out.float()
"""
def Lorentz_layer(w0, wp, g):

    # This block of code redefines 'self' variables from model
    normalize_input = True
    geoboundary = [20, 200, 20, 100]
    fre_low = 0.5
    fre_high = 5
    num_lorentz = 4
    num_spec_point = 300
    w_numpy = np.arange(fre_low, fre_high, (fre_high - fre_low) / num_spec_point)

    # Create the tensor from numpy array
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        w = torch.tensor(w_numpy).cuda()
    else:
        w = torch.tensor(w_numpy)

    w0 = w0.unsqueeze(2)
    wp = wp.unsqueeze(2)
    g = g.unsqueeze(2)

    # Expand them to the make the parallelism, (batch_size, #Lor, #spec_point)
    w0 = w0.expand(w0.size(0), num_lorentz, num_spec_point)
    wp = wp.expand_as(w0)
    g = g.expand_as(w0)
    w_expand = w.expand_as(g)

    # e2 = div(mul(pow(wp, 2), mul(w_expand, g)),
    #          add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))

    num = mul(pow(wp, 2), mul(w_expand, g))
    denom = add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2)))
    # denom = scale_grad.apply(denom)
    e2 = div(num, denom)
    # e2 = mul(add(num, denom), 0.01)


    e2 = torch.sum(e2, 1)
    out = e2.float()
    return out
"""
