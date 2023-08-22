##################################################################################
# Cross-modal Guided Visual Representation Learning for Social Image Retrieval   #
#                                                                                #
#                                                                                #
# Description: This .py includes the loss functions for training the CGVR        #
#                                                                                #
# Note: This code is used for ICCV2023 review purposes                           #
##################################################################################


import os, sys
import os.path as osp
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

# ContrastiveLoss  for CGVR
class ContrastiveLoss(nn.Module):

    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.margin_scaler = args.margin_scaler
        self.ignoring_rate = args.ignoring_rate
        self.log_softmax = F.log_softmax
        self.kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)
        
    def self_contrastive_loss(self, img, tgt, caps_embed, epochs, sim_mat):
        batch_size, _, embed_dim = caps_embed.shape
        sim_mat = self.softmax(sim_mat).unsqueeze(dim=-1).repeat(1, 1, embed_dim)
        caps_embed_mean = torch.mul(sim_mat, caps_embed).sum(dim=1)  # [N, dim]
        cos_caps_embed = calculate_cosine_similarity_matrix(caps_embed_mean, caps_embed_mean).clamp(1e-6, 1.0 - 1e-6)
        scores = torch.mm(img, tgt.T)
        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        cost_s = ((1 - cos_caps_embed) * self.margin_scaler + scores - d1).clamp(min=1e-8)
        I = torch.eye(scores.size(0)) > .5
        I = I.to(self.args.device)
        cost_s = cost_s.masked_fill_(I, 0)

        clean_rate_ = epochs * self.ignoring_rate if epochs * self.ignoring_rate < 1 else 0.99
        k = math.ceil(batch_size * batch_size * (1 - clean_rate_))
        topk = torch.topk(cost_s.flatten(), k, largest=False)
        topk_lossvalue = topk.values[-1]

        cost_s = torch.where(cost_s < topk_lossvalue, cost_s, torch.zeros_like(cost_s))

        return cost_s.sum()
    
    def distillationLoss(self, stu_imgs, tcher_imgs, temperature=0.1):

        p_tcher = self.log_softmax(tcher_imgs / temperature, dim=-1)
        p_stu = self.log_softmax(stu_imgs / temperature, dim=-1)
        loss = self.kl_loss(p_stu, p_tcher)
        return loss
    
    def forward(self, outputs_h1, outputs_h2, caps_embed, sim_mat, epochs, s_v, s_v2):
        loss1 = self.self_contrastive_loss(outputs_h1, outputs_h2, caps_embed, epochs,
                                           sim_mat.detach().squeeze(dim=-1))
        loss2 = quantization_loss(outputs_h1, self.args.CGVR_nbits)
        loss3 = self.distillationLoss(s_v2, s_v)
        awl = AutomaticWeightedLoss(3)  # we have 2 losses
        loss_sum = awl(loss1, loss2, loss3)
        return loss_sum





def quantization_loss(img, nbits, lamda=1.0):
    loss = - lamda * torch.sum((1 / nbits) * torch.pow((img - 0.5), 2), dim=1)
    return loss.mean()


def calculate_cosine_similarity_matrix(h_emb, eps=1e-5):
    # h_emb (N, M)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))
    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix