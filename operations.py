import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  "MB_3x3_3": lambda C, stride, affine: MobileBottleneck(C,C,3,stride,exp=3,nl='RE',affine=affine),
  "MB_3x3_6": lambda C, stride, affine: MobileBottleneck(C,C,3,stride,exp=6,nl='RE',affine=affine),
  "MB_5x5_3": lambda C, stride, affine: MobileBottleneck(C,C,5,stride,exp=3,nl='RE',affine=affine),
  "MB_5x5_6": lambda C, stride, affine: MobileBottleneck(C,C,5,stride,exp=6,nl='RE',affine=affine),
  "MB_7x7_3": lambda C, stride, affine: MobileBottleneck(C,C,7,stride,exp=3,nl='RE',affine=affine),
  "MB_7x7_6": lambda C, stride, affine: MobileBottleneck(C,C,7,stride,exp=6,nl='RE',affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class Hswish(nn.Module):
  def __init__(self, inplace=True):
    super(Hswish, self).__init__()
    self.inplace = inplace

  def forward(self, x):
    return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class hsigmoid(nn.Module):
  def forward(self, x):
    out = F.relu6(x + 3, inplace=True) / 6
    return out

class Hsigmoid(nn.Module):
  def __init__(self, inplace=True):
    super(Hsigmoid, self).__init__()
    self.inplace = inplace

  def forward(self, x):
    return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
  def __init__(self, channel, reduction=4):
    super(SEModule, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      Hsigmoid()
      # nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

class MobileBottleneck(nn.Module):
  def __init__(self, inp, oup, kernel, stride, exp, se=True, nl='RE',affine=True):
    super(MobileBottleneck, self).__init__()
    assert stride in [1, 2]
    assert kernel in [3, 5, 7]
    padding = (kernel - 1) // 2
    self.use_res_connect = stride == 1 and inp == oup

    conv_layer = nn.Conv2d
    norm_layer = nn.BatchNorm2d
    if nl == 'RE':
      nlin_layer = nn.ReLU  # or ReLU6
    elif nl == 'HS':
      nlin_layer = Hswish
    else:
      raise NotImplementedError
    if se:
      SELayer = SEModule
    else:
      SELayer = Identity
    hidden_dim=round(inp*exp)
    self.conv = nn.Sequential(
      # pw
      conv_layer(inp, hidden_dim, 1, 1, 0, bias=False),
      norm_layer(hidden_dim,affine=affine),
      nlin_layer(inplace=True),
      # dw
      conv_layer(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim, bias=False),
      norm_layer(hidden_dim),
      SELayer(hidden_dim),
      nlin_layer(inplace=True),
      # pw-linear
      conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
      norm_layer(oup)
      )

  def forward(self, x):
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)