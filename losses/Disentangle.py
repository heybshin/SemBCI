import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentLoss(nn.Module):
    def __init__(self, estimator, type):
        super().__init__()
        self.estimator = estimator
        self.type = type

    def forward(self, z1, z2):
        Ldis = - self.compute_mutual_info(z1, z2)
        if self.type == 'ortho':
            Ldis += torch.norm(torch.matmul(z1.T, z2)) ** 2
        return Ldis

    def compute_mutual_info(self, z1, z2):
        z1 = z1.view(z1.shape[0], -1)
        z2 = z2.view(z2.shape[0], -1)
        z1_n = torch.index_select(z1, 0, torch.randperm(z1.shape[0]).to(z1.device))

        joint = self.estimator(z2, z1)
        joint = (torch.log(torch.tensor(2.0)) - F.softplus(-joint))
        marginal = self.estimator(z2, z1_n)
        marginal = (F.softplus(-marginal) + marginal - torch.log(torch.tensor(2.0)))
        jsd_loss = joint - marginal

        return jsd_loss.mean()
