import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm

import torch
from torch.utils.tensorboard import SummaryWriter

# t = torch.load('./stats/stat_cifar.pth')
# mask   = np.array([1, 3])
# mu     = t["mu"].detach().numpy()
# S      = np.exp(t["logvar"].detach().numpy())
# rff    = t["rff"] # list

# masked_S  = S[:, mask]
# masked_rff = [rff[m] for m in mask]

t = torch.load('./stats_one_task/stat_cifar100_one_task.pth')
basis, rff, r_mu, r_log_var = t["basis"], t["rff"], t["mu"], t["log_var"]
mu      = torch.cat(r_mu, dim=0)
log_var = torch.cat(r_log_var, dim=0)
mu     = mu.detach().cpu().numpy()
S      = np.exp(log_var.detach().cpu().numpy())

import pdb
pdb.set_trace()

writer = SummaryWriter()

all_feat = []
all_label = []
for task_id, task_rff in enumerate(rff):
    if task_id == 5:
        break
    task_rff = task_rff
    label = [task_id+1 for i in range(task_rff.shape[0])]
    
    all_feat.append(task_rff)
    all_label = all_label + label

# label_txt = [f"{lbl}" for lbl in all_label]
all_feat = torch.cat(all_feat, dim=0)
writer.add_embedding(all_feat, all_label)#, label_img=torch.tensor(all_label))
writer.close()