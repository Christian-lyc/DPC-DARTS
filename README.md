# DPC-DARTS
This is a novel PC-DARTS, whose architecure is deversified by batch normalization. Our results show the number of parameters can be reduced 40% compared with DARTS.

Abstract:

The differentiable architecture search (DARTS) has greatly improved the efficiency of neural architecture search by applying gradient-based optimization. However, both the normal cells and reduction cells in the structure are sharing the same cell structure. This
kind of single structure destroys the diversity of DARTS. Hence, this paper is proposed to address this issue by introducing channel-wise batch normalization. We propose a novel method which helps to extremely reduce number of parameters of the network and as a side effect improves the training stability with faster network convergence and lower memory consumption in comparison to the original DARTS. The conducted experimental based on CIFAR10 and CIFAR100 data sets reveal high performance compared to state of art methods.
