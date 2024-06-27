import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from utility.rohe_gat import GATConv
import numpy as np
from utility.Utils import (get_setting, get_hete_adjs, get_transition)


# Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        '''
        Shape of z: (N, M , D*K)
        N: number of nodes
        M: number of metapath patterns
        D: hidden_size
        K: number of heads
        '''
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout, settings):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    settings=settings[i],
                )
                                   )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[meta_path_pattern] = dgl.metapath_reachable_graph(
                    g, meta_path_pattern)
        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class MHGAN(nn.Module):
    def __init__(self, dataset, g, meta_path_patterns,userkey, itemkey, in_size, out_size, num_heads, dropout, device):
        super(MHGAN, self).__init__()
        self.initializer = nn.init.xavier_uniform_
        self.dataset = dataset
        self.g = g
        self.userkey = userkey
        self.itemkey = itemkey
        self.unum = self.g.num_nodes(userkey)
        self.inum = self.g.num_nodes(itemkey)
        self.device = device
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

        self.meta_path_patterns = meta_path_patterns

        self.hans = nn.ModuleDict()
        for key, meta_path in self.meta_path_patterns.items():
            given_hete_adjs = get_hete_adjs(g, meta_path, self.device)
            trans_adj_list = get_transition(given_hete_adjs, meta_path)
            settings = get_setting(self.dataset, key, trans_adj_list, self.device)
            self.hans.update({key: HANLayer(meta_path, in_size, out_size, num_heads, dropout, settings)})

        self.user_layer1 = nn.Linear(out_size, out_size, bias=True)
        self.item_layer1 = nn.Linear(out_size, out_size, bias=True)

        # layernorm
        self.layernorm = nn.LayerNorm(out_size)

        # network to score the node pairs
        self.pred = nn.Linear(out_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_size, 1)


    def forward(self, user_idx, item_idx, neg_item_idx):
        user_key = self.userkey
        item_key = self.itemkey

        h2 = {}
        for key in self.meta_path_patterns.keys():
            h2[key] = self.hans[key](self.g, self.feature_dict[key])

        user_emb = h2[user_key]
        item_emb = h2[item_key]
        user_emb = self.user_layer1(user_emb)
        item_emb = self.item_layer1(item_emb)

        # # Relu
        user_emb = F.relu_(user_emb)
        item_emb = F.relu_(item_emb)

        # layer norm
        user_emb = self.layernorm(user_emb)
        item_emb = self.layernorm(item_emb)

        user_feat = user_emb[user_idx]
        item_feat = item_emb[item_idx]
        neg_item_feat = item_emb[neg_item_idx]

        return user_feat, item_feat, neg_item_feat

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb = self.forward(users, pos, neg)

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def ssl_loss(self, data1, data2, index):
        index=torch.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = torch.exp(pos_score / 0.5)
        all_score  = torch.sum(torch.exp(all_score / 0.5), dim = 1)
        ssl_loss  = (-torch.sum(torch.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss

    def getUsersRating(self, user_idx):
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        rating = torch.matmul(users_emb, all_items.t())
        return rating
