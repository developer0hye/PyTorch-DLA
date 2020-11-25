import os, sys

import math
import torch
import torch.nn as nn
import numpy as np


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size//2,
                      bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))

def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size//2,
                      bias=False), 
            nn.BatchNorm2d(out_channels))
    
class Aggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aggregation, self).__init__()
        self.aggregation = conv_bn_relu(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.aggregation(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.MaxPool2d(2),
                                          conv_bn(in_channels, out_channels, kernel_size=1))
        
        self.conv1 = conv_bn_relu(in_channels, out_channels, stride=stride)
        self.conv2 = conv_bn(out_channels, out_channels)
        
    def forward(self, x):
        return torch.relu(self.shortcut(x) + self.conv2(self.conv1(x)))

class HDAHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block):
        super(HDAHead, self).__init__()
        
        self.block1 = block(in_channels, out_channels, stride=2)
        self.block2 = block(out_channels, out_channels)
        self.aggregation = Aggregation(out_channels * 2, out_channels)
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x = self.aggregation([x1, x2])
        return x

class HDATail(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block):
        super(HDATail, self).__init__()
        
        self.downsample = nn.MaxPool2d(2)
        self.block1 = block(in_channels, out_channels, stride=2)
        self.block2 = block(out_channels, out_channels)
        self.aggregation = Aggregation(in_channels + out_channels * 2, out_channels)
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x = self.aggregation([self.downsample(x), x1, x2])
        return x

class HDABody(nn.Module):       
    def __init__(self,
                 in_channels,
                 out_channels,
                 depth,
                 block):
        super(HDABody, self).__init__()
        
        self.downsample = nn.MaxPool2d(2)
        self.tree = nn.Identity()
        self.aggregation = Aggregation(in_channels + out_channels * depth, out_channels)
        
    def forward(self, x):
        x_downsampled = self.downsample(x) # ch: in_channels
        x = self.tree(x) # x is list that has N(=self.depth) items. ch: out_channels * depth
        x = self.aggregation([x_downsampled, x])
        return x

class DLA(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 num_classes=1000,
                 block=BasicBlock,
                 residual_root=False,
                 pool_size=7,
                 linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels

        #Refer to 4.1. Classification Networks in paper "Deep Layer Aggregation"
        
        #"basic block" means conv_bn_relu layer in stage1 and stage2 and means one of three types of residual blocks in all other stages.
        
        #The stage 1 is composed of a 7×7 convolution followed by a basic block.
        self.stage1 = nn.Sequential(conv_bn_relu(3, channels[0], kernel_size=7),
                                    conv_bn_relu(channels[0], channels[0]))
        
        #The stage 2 is only a basic block.
        self.stage2 = conv_bn_relu(channels[0], channels[1], stride=2)
        
        #For all other stages, we make use of combined IDA and HDA on the backbone blocks and stages.
        
        #To simplfy the code, I constraint the depth of stage3 to 1.
        self.stage3 = HDAHead(channels[1], channels[2], block)
        
        self.stage4 = HDATail(channels[2], channels[3], block)
        self.stage5 = HDATail(channels[3], channels[4], block)
        
        #To simplfy the code, I constraint the depth of stage6 to 1.
        self.stage6 = HDATail(channels[4], channels[5], block)
       
        self.stages = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.stage6])
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))#global average pooling

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        y = []
        
        for stage in self.stages:
            x = stage(x)
            y.append(x)
        
        x = self.gap(x)
        
        return x
    
if __name__ == '__main__':
    
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    
    # base_name = 'dla34'

    # x = torch.randn(1, 3, 256, 256)
    # model = globals()[base_name](pretrained=True, return_levels=False)
    

    # 모델 변환
    torch.onnx.export(model,               # 실행될 모델
                    x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                    "dla34_yh.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                    export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                    opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                    do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                    input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                    output_names = ['output'], # 모델의 출력값을 가리키는 이름
                    dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                  'output' : {0 : 'batch_size'}})
    
    
    
    


# class HDATree(nn.Module):
#     def __init__(self,
#                  depth,
#                  block,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  is_root=False):
#         super(HDATree, self).__init__()
        
#         if depth == 1:
            
#             self.tree1 = block(in_channels,
#                                out_channels,
#                                stride,
#                                dilation=dilation)
            
#             self.tree2 = block(out_channels,
#                                out_channels,
#                                1,
#                                dilation=dilation)
        
        

# class Tree(nn.Module):
#     def __init__(self,
#                  levels,
#                  block,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  level_root=False,
#                  root_dim=0,
#                  root_kernel_size=1,
#                  dilation=1,
#                  root_residual=False):
#         super(Tree, self).__init__()
#         if root_dim == 0:
#             root_dim = 2 * out_channels
#         if level_root:
#             root_dim += in_channels
#         if levels == 1:
#             self.tree1 = block(in_channels,
#                                out_channels,
#                                stride,
#                                dilation=dilation)
#             self.tree2 = block(out_channels,
#                                out_channels,
#                                1,
#                                dilation=dilation)
#         else:
#             self.tree1 = Tree(levels - 1,
#                               block,
#                               in_channels,
#                               out_channels,
#                               stride,
#                               root_dim=0,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation,
#                               root_residual=root_residual)
#             self.tree2 = Tree(levels - 1,
#                               block,
#                               out_channels,
#                               out_channels,
#                               root_dim=root_dim + out_channels,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation,
#                               root_residual=root_residual)
#         if levels == 1:
#             self.root = Root(root_dim, out_channels, root_kernel_size,
#                              root_residual)
#         self.level_root = level_root
#         self.root_dim = root_dim
#         self.downsample = None
#         self.project = None
#         self.levels = levels
#         if stride > 1:
#             self.downsample = nn.MaxPool2d(stride, stride=stride)
#         if in_channels != out_channels:
#             self.project = nn.Sequential(
#                 nn.Conv2d(in_channels,
#                           out_channels,
#                           kernel_size=1,
#                           stride=1,
#                           bias=False), nn.BatchNorm2d(out_channels))

#     def forward(self, x, residual=None, children=None):
#         children = [] if children is None else children
#         bottom = self.downsample(x) if self.downsample else x
#         residual = self.project(bottom) if self.project else bottom
#         if self.level_root:
#             children.append(bottom)
#         x1 = self.tree1(x, residual)
#         if self.levels == 1:
#             x2 = self.tree2(x1)
#             x = self.root(x2, x1, *children)
#         else:
#             children.append(x1)
#             x = self.tree2(x1, children=children)
#         return x
