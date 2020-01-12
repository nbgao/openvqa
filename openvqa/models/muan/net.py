# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.muan.mua import UnifiedLayers
from openvqa.models.muan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import en_vectors_web_lg

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class TokenFlat(nn.Module):
    def __init__(self, token_pos=0):
        super(TokenFlat, self).__init__()
        self.token_pos = token_pos
    
    def forward(self, x, x_mask):
        return x[:, self.token_pos, :]

'''
class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
'''

# -------------------------
# ---- Main MUAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # spacy_tool = en_vectors_web_lg.load()
        # cls_vector = np.expand_dims(spacy_tool('CLS').vector, axis=0)
        # pretrained_emb = np.concatenate((cls_vector, pretrained_emb), axis=0)

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.backbone = UnifiedLayers(__C)

        # Flatten to vector
        self.flat = TokenFlat(token_pos=0)

        # Classification layers
        # self.proj_norm = LayerNorm(__C.HIDDEN_SIZE)
        self.proj = nn.Linear(__C.HIDDEN_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        cls_tensor = torch.full((ques_ix.shape[0], 1), 2, dtype=torch.long).cuda()
        ques_ix = torch.cat((cls_tensor, ques_ix), dim=1)

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        lang_feat = self.norm1(lang_feat)
        img_feat = self.norm2(img_feat)

        fuse_feat = torch.cat((lang_feat, img_feat), dim=1)
        fuse_feat_mask = torch.cat((lang_feat_mask, img_feat_mask), dim=-1)

        # Backbone Framework
        fuse_feat = self.backbone(fuse_feat, fuse_feat_mask)

        # Flatten to vector
        fuse_flat = self.flat(
            fuse_feat,
            fuse_feat_mask
        )

        # Classification layers
        # proj_feat = lang_feat + img_feat
        # proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(fuse_flat)

        return proj_feat

