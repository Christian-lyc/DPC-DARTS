import torch
import torch.nn as nn
import argparse
from train_search import Network
import numpy as np


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
args = parser.parse_args()
CIFAR_CLASSES=100
PATH='/home/yichao/PC-DARTS-master/search-EXP-20191025-092118/weights.pt'
# PATH='/home/yichao/Desktop/darts-master/cnn/search-EXP-20190829-102303/weights.pt'
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()



model=Network(args.init_channels, CIFAR_CLASSES,args.layers, criterion)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(PATH))
model.eval()



select_op=[]

index = 0

states=2
PX=[]
Op=model.module.cells._modules.items()
for m in Op:
    for n in m[1]._ops._modules.items():

        if isinstance(n[1]._bn, nn.BatchNorm2d):
            size = n[1]._bn.weight.data.shape[0]
            Y=list(n[1]._bn.weight.data.abs().chunk(len(n[1]._ops)))
            Y=torch.cat([j.view(1,-1) for j in Y])  #view func transfer 1 dim tensor to 2 dim, then cat it
            weight_sum=torch.sum(Y,dim=1)
            select_op.append(torch.max(weight_sum,0)[1])
    for k in m[1]._bns:
        if isinstance(k,nn.BatchNorm2d):
            Z=[torch.mean(k.weight.data.abs().chunk(states, dim=0)[j]) for j in range(states)]
            PX.append(np.argsort(Z)[-2:])
            print(Z)
            print(np.sort(np.asarray(PX)).tolist()) #asarray Convert the input to an array.

            states=states+1
            if states==6:
                states=2


select_oper=[]
for i in select_op:
    select_oper.append(i.item())

print(select_oper)



