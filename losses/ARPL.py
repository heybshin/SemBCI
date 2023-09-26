"""
Adapted from G. Chen, P. Peng, X. Wang, and Y. Tian, "Adversarial reciprocal points learning for open set recognition,"
IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 8065–8081, Apr. 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist


class ARPLoss(nn.CrossEntropyLoss):
    """Adversarial Reciprocal Points Learning (ARPL) Loss
    # Adapted from G. Chen, P. Peng, X. Wang, and Y. Tian, "Adversarial reciprocal points learning for open set recognition,"
# IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 8065–8081, Apr. 2021.

    """

    def __init__(self, n_classes=10, latent_dim=128, arpl_alpha=0.1, cls_alpha=1.0, temp=1.0):
        super(ARPLoss, self).__init__()
        self.arpl_alpha = float(arpl_alpha)
        self.cls_alpha = float(cls_alpha)
        self.temp = temp
        self.Dist = Dist(num_classes=n_classes, feat_dim=latent_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.loss_margin = nn.MarginRankingLoss(margin=1.0)
        self.loss_cls = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')   # dot product between each instance and class center
        dist_l2_p = self.Dist(x, center=self.points, metric='l2')  # l2 distance between each instance and class center
        logits = dist_l2_p - dist_dot_p
        # loss = F.cross_entropy(logits / self.temp, labels)
        loss = self.loss_cls(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.loss_margin(self.radius, _dis_known, target)

        loss = self.cls_alpha * loss + self.arpl_alpha * loss_r

        return loss, logits

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss