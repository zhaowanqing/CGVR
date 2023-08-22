##################################################################################
# Cross-modal Guided Visual Representation Learning for Social Image Retrieval   #
#                                                                                #
#                                                                                #
# Description: This .py is the implementation of our Relational-enhanced         #
#              Cross-modal Representation Network                                #
#                                                                                #
# Note: This code is used for ICCV2023 review purposes                           #
##################################################################################



import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class Relational_Attention(nn.Module):

    def __init__(self, kg_attention_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(kg_attention_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                KG_matrix):
        output = src

        for layer in self.layers:
            output = layer(output, KG_matrix)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Relational_Attention_Layer(nn.Module):
    def __init__(self, d_model, KG_dim, dropout=0.1, activation="relu"):
        super().__init__()
        self.fc = torch.nn.Linear(2 * d_model + KG_dim, 1)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.norm = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model, bias=True)
        self.linear_k = nn.Linear(d_model, d_model, bias=True)
        self.linear_v = nn.Linear(d_model, d_model, bias=True)
        self.linear_kg = nn.Linear(KG_dim, KG_dim, bias=True)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_model, bias=True)
        self.linear2 = nn.Linear(d_model, d_model, bias=True)

    def forward(self, vectors, KG_matrix):
        bsz, n_words, dim = vectors.shape

        q, k, v, kg = self.linear_q(vectors), self.linear_k(vectors), self.linear_v(vectors), self.linear_kg(KG_matrix)
        q = self.activation(q)
        k = self.activation(k)
        v = self.activation(v)
        kg = self.activation(kg)

        # q, k, v, kg = vectors, vectors, vectors, KG_matrix
        q = q.repeat(1, 1, n_words).reshape(bsz, n_words, n_words, dim)
        k = k.transpose(1, 2).repeat(1, 1, n_words).transpose(1, 2).reshape(bsz, n_words, n_words, dim)

        z = torch.cat([q, k, kg], dim=-1)
        attn_weights = self.leaky_relu(self.fc(z).squeeze(dim=-1))
        attn_weights = torch.softmax(attn_weights, dim=-1) 
        attn_output = torch.bmm(attn_weights, v)

        src = v + self.dropout(attn_output)
        src2 = self.norm(src)
        src2 = self.linear2(self.dropout1(self.leaky_relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class cross_modal_module(nn.Module):

    @staticmethod
    def weight_init(m):
       if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 num_kg_attention_layer=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", use_KG=True):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        self.use_KG = use_KG

        if num_encoder_layers > 0:
            encoder_layer_txt = TransformerEncoderLayer(d_model, nhead, d_model, dropout, activation)
            encoder_norm_txt = torch.nn.LayerNorm(d_model)
            self.encoder_txt = TransformerEncoder(encoder_layer_txt, num_encoder_layers, encoder_norm_txt)


        if num_kg_attention_layer > 0:
            kg_attention_layer_txt = Relational_Attention_Layer(d_model, 50)
            norm_txt = torch.nn.LayerNorm(d_model)
            self.kg_attention_txt = Relational_Attention(kg_attention_layer_txt, num_kg_attention_layer, norm_txt)

        if num_kg_attention_layer > 0:
            kg_attention_layer_img = Relational_Attention_Layer(d_model, 50)
            norm_img = torch.nn.LayerNorm(d_model)
            self.kg_attention_img = Relational_Attention(kg_attention_layer_img, num_kg_attention_layer, norm_img)
            
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_model, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        decoder_layer2 = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder2 = TransformerDecoder(decoder_layer2, num_decoder_layers, decoder_norm2)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_feat, txt_embed, pos_embed, pad_mask=None, KG_matrix=None):

        #src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        img_feat = img_feat.permute(1, 0, 2)    
        if self.num_encoder_layers > 0:
            tgt = self.encoder_txt(txt_embed, src_key_padding_mask = pad_mask.permute(1, 0))
        else:
            tgt = txt_embed
        if self.use_KG:
            tgt = self.kg_attention_txt(tgt, KG_matrix).permute(1, 0, 2)
        else:
            tgt = tgt.permute(1, 0, 2)

        v = self.decoder(tgt, img_feat, tgt_key_padding_mask=pad_mask, pos=pos_embed, query_pos=None, tgt_mask=None)
        
        if self.use_KG:
            v_hat = self.kg_attention_img(v.squeeze(dim=0).transpose(0, 1), KG_matrix).transpose(0, 1)
        else:
            v_hat = v.squeeze(dim=0)
        
        return tgt.permute(1, 0, 2), v_hat.permute(1, 0, 2)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2, sim_mat_1 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                         key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, sim_mat_2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                              key=self.with_pos_embed(memory, pos),
                                              value=memory, attn_mask=memory_mask,
                                              key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_cross_modal_module(args):
    return cross_modal_module(
        d_model=args.d_model,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_kg_attention_layer=args.kg_layers,
        dim_feedforward=args.d_model,
        dropout=args.dropout,
        use_KG=args.use_KG
    )
