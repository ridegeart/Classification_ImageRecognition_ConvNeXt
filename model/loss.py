import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import nn, Tensor

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	        (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''
    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)
        # mat_dist = cosine_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0)==dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class CenterLoss(nn.Module):
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

