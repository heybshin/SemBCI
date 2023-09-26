
import torch
import torch.nn as nn


class Expand(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.unsqueeze(self.axis)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) that reverses the gradient during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class DCN(nn.Module):  # deep convnet encoder

    def conv1(self, outF, kernalSize, nchans, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(outF, outF, (nchans, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, self.k_pool], stride=[1, self.s_pool])
        )

    def conv2(self, inF, outF, do, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=do),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, self.k_pool], stride=[1, self.s_pool])
        )

    def compute_nfeats(self, model, nchans, ntimes):
        x = torch.rand(1, nchans, ntimes)
        model.eval()
        return model(x).shape[-1]

    def __init__(self, nchans=64, ntimes=1001, do=0):
        super(DCN, self).__init__()
        k = (1, 5)
        n_filt_first = 25
        n_filts = [25, 50, 50, 100]
        self.k_pool = 7  # int(sfreq * 0.1)
        self.s_pool = 3  # int(sfreq * 0.1 * (1 - 0.7))

        first_layer = nn.Sequential(
            Expand(1),
            self.conv1(n_filt_first, k, nchans)
        )
        middle_layers = nn.Sequential(*[self.conv2(inF, outF, do, k) for inF, outF in zip(n_filts, n_filts[1:])])
        self.conv_block = nn.Sequential(
            first_layer,
            middle_layers,
            nn.Flatten()
        )

        self.nfeats = self.compute_nfeats(self.conv_block, nchans, ntimes)

    def forward(self, x):
        return self.conv_block(x)  # [1, 25, 1, 249] #new: (B, 500)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nchans, ntimes = cfg.DATA.n_chans, cfg.DATA.n_times
        ldim = cfg.MODEL.latent_dim
        self.feature_extractor = DCN(nchans, ntimes, do=cfg.MODEL.dropout)
        nfeats = self.feature_extractor.nfeats
        if cfg.DISENT:
            self.rele_encoder = nn.Sequential(
                nn.Linear(nfeats, ldim),
                nn.GroupNorm(4, ldim),
                nn.PReLU()
            )
            self.irre_encoder = nn.Sequential(
                nn.Linear(nfeats, ldim),
                nn.GroupNorm(4, ldim),
                nn.PReLU()
            )
        else:
            self.encoder = nn.Linear(nfeats, ldim)

    def forward(self, x):
        x = self.feature_extractor(x)
        if hasattr(self, 'rele_encoder'):
            z = self.rele_encoder(x)
            zi = self.irre_encoder(x)
            return z, zi
        z = self.encoder(x)
        return z


class MIEstimator(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) class that estimates the mutual information
    between two latent features. Employs Gradient Reversal Layer and concatenation-based method.
    """
    def __init__(self, latent_dim):
        super(MIEstimator, self).__init__()

        hidden_dim = latent_dim // 16

        self.parallel = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout()
        )

        self.combined = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU(),
            nn.Linear(hidden_dim*2, 1)
        )

    def forward(self, z1, z2, lambd=1.0):
        z1, z2 = grad_reverse(z1, lambd), grad_reverse(z2, lambd)
        z1, z2 = self.parallel(z1), self.parallel(z2)
        return self.combined(torch.cat((z1, z2), dim=-1))


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=hidden_dim, out_features=emb_dim)
            )
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_out(x)
        return x