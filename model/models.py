##################################################################################
# Cross-modal Guided Visual Representation Learning for Social Image Retrieval   #
#                                                                                #
#                                                                                #
# Description: This .py includes the implementation of our CGVR                  #
#                                                                                #
# Note: This code is used for ICCV2023 review purposes                           #
##################################################################################



import torch
import torch.nn as nn
import math
from model.cross_modal_module import build_cross_modal_module
from model.backbone import build_backbone

def create_rel_vector2(targets, lengths, rel_matrix):
    B, N = targets.shape[0], targets.shape[1]
    KG_Matrix = torch.zeros(B, N, N, 40)
    for i in range(B):
        len_i = lengths[i]
        targets_i = targets[i]
        for j, index_c1 in enumerate(targets_i[:len_i]):
            for k, index_c2 in enumerate(targets_i[:len_i]):
                rel = int(rel_matrix[index_c1-4][index_c2-4].item())
                if(rel != -1):
                    KG_Matrix[i][k][j][rel] = 1
    return KG_Matrix

def create_rel_vector(targets, lengths, rel_matrix):
    B, N = targets.shape[0], targets.shape[1]
    KG_Matrix = torch.zeros(B, N, N, 50)
    for i in range(B):
        len_i = lengths[i]
        targets_i = targets[i]
        for j, index_c1 in enumerate(targets_i[:len_i]):
            for k, index_c2 in enumerate(targets_i[:len_i]):
                KG_Matrix[i][j][k] = rel_matrix[index_c1-4][index_c2-4]
    return KG_Matrix

def create_attn_mask(targets, lengths, relatnessWeight):
    B, N = targets.shape[0], targets.shape[1]
    attn_mask = torch.ones(B, N, N)
    for i in range(B):
        len_i = lengths[i]
        targets_i = targets[i]
        for j, index_c1 in enumerate(targets_i[:len_i]):
            for k, index_c2 in enumerate(targets_i[:len_i]):
                attn_mask[i][j][k] = relatnessWeight[index_c1-4][index_c2-4]
    return attn_mask

#Cross-modal Guided Visual Representation Learning (CGVR)
class CGVR(nn.Module):
    def __init__(self, backbone, cross_modal_module, word_vectors, rel_matrix, args):
        super().__init__()
        self.backbone = backbone
        self.cross_modal_module = cross_modal_module
        self.args = args
        self.nheads = args.nheads
        self.rel_matrix = rel_matrix

        self.txt_embed = nn.Embedding.from_pretrained(word_vectors)
        self.txt_embed.weight.requires_grad = False
        self.linear_caps_embed = nn.Linear(args.word_dim, args.d_model)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.nbits = args.CGVR_nbits
        self.hash_fc_img = nn.Linear(args.d_model, args.CGVR_nbits)
        self.hash_fc_tgt = nn.Linear(args.d_model, args.CGVR_nbits)

        self.W1 = nn.Linear(args.d_model, 1, bias=False)
        self.W2 = nn.Linear(args.d_model, 1, bias=False)  
        self.W3 = nn.Linear(args.d_model, 1, bias=False)  
        if args.enc_layers > 0:
            encoder_layer_img = torch.nn.TransformerEncoderLayer(self.backbone.num_channels, args.nheads, args.d_model)
            encoder_norm_img = torch.nn.LayerNorm(self.backbone.num_channels)
            self.encoder_img = torch.nn.TransformerEncoder(encoder_layer_img, args.enc_layers, encoder_norm_img)

        
    def forward(self, input, targets = None, lengths = None):
        img_feat, pos = self.backbone(input)
        img_feat, pos = img_feat[-1], pos[-1]  # (256x2048x7x7) (256x2048x7x7)        
        out_tgt, out_img, w_vector, s_v, s_v2, s_t = None, None, None, None, None, None
        img_feat = img_feat.flatten(2).permute(0, 2, 1)
        


        if self.training:
            txt_pad_mask = targets.data.eq(0)  # （N, n_words）
            # Knowledge relational type matrix 
            KG_matrix = create_rel_vector(targets, lengths, self.rel_matrix).to(self.args.device)
        
            w_vector = self.txt_embed(targets).to(torch.float)
            w = self.linear_caps_embed(w_vector)

            # Knowledge-Enhanced Cross-Modal Awareness module
            tat, hs = self.cross_modal_module(img_feat, w, pos_embed=pos, pad_mask=txt_pad_mask,
                                           KG_matrix=KG_matrix)
       
            attention_scores = torch.matmul(img_feat, hs.permute(0, 2, 1))/ math.sqrt(img_feat.shape[-1])
            attention_scores = torch.softmax(attention_scores, dim=-1) 
            temp_img = torch.matmul(attention_scores, hs)
            temp_tgt = tat + hs
            # s_v, s_v2: attention for visual feature,  s_t: attention for tag embedding
            s_t = self.softmax(self.W2(self.tanh(temp_tgt)).squeeze()).unsqueeze(dim=1)  # (N, n_words, 1)
            s_v2 = self.softmax(self.W1(img_feat).squeeze()).unsqueeze(dim=1)  # (N, 49, 1)   
            s_v = self.softmax(self.W1(temp_img).squeeze()).unsqueeze(dim=1)  # (N, 49, 1)   
 
            out_img = (s_v @ img_feat).squeeze()         
            
 
            out_tgt  = (s_t @ w ).squeeze()
            out_tgt = self.sigmoid(self.hash_fc_tgt(out_tgt))
            out_img = self.sigmoid(self.hash_fc_img(out_img))
            return out_img, out_tgt, w_vector, s_v.squeeze(), s_t.squeeze(), s_v2.squeeze()
        else:
            
            s_v = self.softmax(self.W1(img_feat).squeeze()).unsqueeze(dim=1)  # (N, 49, 1)        
            out_img = (s_v @ img_feat).squeeze() 

            out_img = self.sigmoid(self.hash_fc_img(out_img))
            return out_img



def build_CGVR_net(word_vectors, rel_matrix, args):

    cross_modal_module = build_cross_modal_module(args)
    backbone = build_backbone(args.CGVR_backbone,args)
    model = CGVR(
        backbone=backbone,
        cross_modal_module=cross_modal_module,
        word_vectors=word_vectors,
        rel_matrix=rel_matrix,
        args=args
    )
    return model
