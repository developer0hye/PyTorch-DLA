import os, sys

import math
import torch
import torch.nn as nn
import numpy as np


# model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
#                 block=BasicBlock,
#                 **kwargs)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size//2,
                      bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

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
        
        self.stage1 = nn.Sequential(conv_bn_relu(3, channels[0], kernel_size=7),
                                    conv_bn_relu(channels[0], channels[0]))
        
        self.stage2 = conv_bn_relu(channels[0], channels[1], stride=2)
        
        
        self.stages = nn.ModuleList([self.stage1, self.stage2])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = []
        
        for stage in self.stages:
            x = stage(x)
            y.append(x)
        
        return x
    
if __name__ == '__main__':
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],)
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
