import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
from genotypes import PRIMITIVES


class Cell(nn.Module):

  def __init__(self, genotype, genotype_nodes,C_prev_prev, C_prev, C, reduction, reduction_prev,steps):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self._steps = steps
    self._genotype = genotype
    self._genotype_nodes=genotype_nodes

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    # if reduction:
    #   op_names, indices = zip(*genotype.reduce)
    #   concat = genotype.reduce_concat
    # else:
    #   op_names, indices = zip(*genotype.normal)
    #   concat = genotype.normal_concat
    # self._compile(C, op_names, indices, concat, reduction)
    self._compile(C, steps,reduction,genotype,genotype_nodes)

  # def _compile(self, C, op_names, indices, concat, reduction):
  def _compile(self, C,steps,  reduction,genotype,genotype_nodes):
    # assert len(op_names) == len(indices)
    # self._steps = len(op_names) // 2
    # self._concat = concat
    # self.multiplier = len(concat)
    #
    # self._ops = nn.ModuleList()
    # for name, index in zip(op_names, indices):
    #   stride = 2 if reduction and index < 2 else 1
    #   op = OPS[name](C, stride, True)
    #   self._ops += [op]
    # self._indices = indices

    # self._ops = nn.ModuleList()
    # for ge in range(len(genotype)):
    #     if reduction and ge < 4:
    #       stride = 2
    #     else:
    #       stride =1
    #     op = OPS[PRIMITIVES[genotype[ge]]](C, stride, True)
    #     self._ops.append(op)

    self._ops = nn.ModuleList()

    offset = 0
    # for j in range(len(genotype)):
    for i in range(len(genotype_nodes)):

      for j in genotype_nodes[i]:
        if reduction and j<2:
          stride = 2
        else:
          stride = 1
        op = OPS[PRIMITIVES[genotype[j+offset]]](C,stride,True)
        self._ops.append(op)
      if i==0:
        offset=2
      elif i==1:
        offset=5
      elif i==2:
        offset=9


  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    j=0
    for i in self._genotype_nodes:
      h1 = states[i[0]]
      h2 = states[i[1]]
      op1 = self._ops[2*j]
      op2 = self._ops[2*j+1]
      h1 = op1(h1)
      h2 = op2(h2)

      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      # if i==0:
      #   h1=states[0]
      #   h2=states[1]
      #   op1=self._ops[0]
      #   op2=self._ops[1]
      #   h1=op1(h1)
      #   h2=op2(h2)
        # s = h1 + h2
      # if i==1:
      #   h1 = states[0]
      #   h2 = states[1]
      #   h3 = states[2]
      #   op1 = self._ops[2]
      #   op2 = self._ops[3]
      #   op3 = self._ops[4]
      #   h1=op1(h1)
      #   h2=op2(h2)
      #   h3=op3(h3)
      #   s=h1+h2+h3
      # if i==2:
      #   h1 = states[0]
      #   h2 = states[1]
      #   h3 = states[2]
      #   h4 = states[3]
      #   op1 = self._ops[5]
      #   op2 = self._ops[6]
      #   op3 = self._ops[7]
      #   op4 = self._ops[8]
      #   h1 = op1(h1)
      #   h2 = op2(h2)
      #   h3 = op3(h3)
      #   h4 = op4(h4)
      #   s=h1+h2+h3+h4
      states += [s]
      j = j+1
    # return torch.cat([states[i] for i in self._concat], dim=1)
    return torch.cat([states[i + 2] for i in range(self._steps)], dim=1)

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype,genotype_nodes,steps=4,multiplier=4):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    self._steps = steps
    self._multiplier = multiplier
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3,2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      subgenotype=genotype[i*14:(i+1)*14]
      subgenotype_nodes=genotype_nodes[i*4:(i+1)*4]

      cell = Cell(subgenotype, subgenotype_nodes,C_prev_prev, C_prev, C_curr, reduction, reduction_prev,steps)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr
      # if i==0 or i==2:
      #   cell = Cell(subgenotype, subgenotype_nodes, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, steps)
      #   reduction_prev = reduction
      #   self.cells += [cell]
      #   C_prev_prev, C_prev = C_prev, multiplier*C_curr

      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # print(i)
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype,genotype_nodes,steps=4,multiplier=4):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary


    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      subgenotype = genotype[i * 14:(i + 1) * 14]
      subgenotype_nodes = genotype_nodes[i * 4:(i + 1) * 4]
      cell = Cell(subgenotype,subgenotype_nodes, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,steps)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

