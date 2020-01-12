# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

        self.gated_dot_product = GatedDotProduct(__C, dropout_ratio=0)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        gate = self.gated_dot_product(key, query)
        key = gate[:, :, :, 0:1] * key
        query = gate[:, :, :, 1:2] * query

        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Gated Dot-product ----
# ---------------------------

class GatedDotProduct(nn.Module):
    def __init__(self, __C, dropout_ratio=0):
        super(GatedDotProduct, self).__init__()
        self.__C = __C
        self.dropout_ratio = dropout_ratio
        d_base = int(__C.HIDDEN_SIZE / __C.MULTI_HEAD)
        
        self.linearX = nn.Linear(d_base, d_base)
        self.linearY = nn.Linear(d_base, d_base)
        self.linear = nn.Linear(d_base, 2)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, key, query):
        key = self.linearX(key)
        query = self.linearY(query)
        gate = key * query
        
        if self.dropout_ratio > 0:
            gate = self.dropout(gate)
        
        gate = self.linear(gate)
        gate = F.sigmoid(gate)
        
        return gate


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------
# ---- Unified Attention Block ----
# ---------------------------------

class UA_Block(nn.Module):
    def __init__(self, __C):
        super(UA_Block, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# ------------------------
# ---- Unified Layers ----
# ------------------------

class UnifiedLayers(nn.Module):
    def __init__(self, __C):
        super(UnifiedLayers, self).__init__()
        self.ua_block_list = nn.ModuleList([UA_Block(__C) for _ in range(__C.LAYER)])

    def forward(self, x, mask):
        for ua_block in self.ua_block_list:
            x = ua_block(x, mask)
        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

# class MCA_ED(nn.Module):
#     def __init__(self, __C):
#         super(MCA_ED, self).__init__()

#         self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
#         self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

#     def forward(self, y, x, y_mask, x_mask):
#         # Get encoder last hidden vector
#         for enc in self.enc_list:
#             y = enc(y, y_mask)

#         # Input encoder last hidden vector
#         # And obtain decoder last hidden vectors
#         for dec in self.dec_list:
#             x = dec(x, y, x_mask, y_mask)

#         return y, x
