import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import softmax

model_hyperparam = {
    'HSimplE': {
        'hidden_drop': 0.2,
    }
}



class HSimplE(nn.Module):
    def __init__(self, args):
        super(HSimplE, self).__init__()
        self.emb_dim = args.emb_dim
        self.max_arity = 6

        self.hidden_drop_rate = model_hyperparam[args.kg_model]["hidden_drop"]

        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def forward(self, r, E, ms, bs):
        '''
        r: relation embedding
        E: entity embedding (each row is an entity embedding, containing |r| rows)
        '''

        for i in range(E.shape[1]):
            e = self.shift(E[:, i], int((i + 1) * self.emb_dim / self.max_arity)) * ms[:, i].view(-1, 1) + bs[:,
                                                                                                           i].view(-1,
                                                                                                                   1)
            if i == 0:
                x = e
            else:
                x = x * e
        x = x * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x


class CEGAT(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 heads,
                 output_heads,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            if num_layers == 1:
                self.convs.append(GATConv(in_dim, out_dim, output_heads))
            else:
                self.convs.append(GATConv(in_dim, hid_dim, heads))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(heads * hid_dim, hid_dim, heads))
                    self.normalizations.append(nn.BatchNorm1d(hid_dim))

                self.convs.append(GATConv(heads * hid_dim, out_dim,
                                          heads=output_heads, concat=False))
        else:  # default no normalizations
            if num_layers == 1:
                self.convs.append(GATConv(in_dim, out_dim, output_heads))
            else:
                self.convs.append(GATConv(in_dim, hid_dim, heads))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(hid_dim * heads, hid_dim, heads))
                    self.normalizations.append(nn.Identity())

                self.convs.append(GATConv(hid_dim * heads, out_dim,
                                          heads=output_heads, concat=False))

        self.dropout = dropout

    def forward(self, x, edges, training=True):
        #        x: [N, in_dim] (all vertices)
        #        edge_index: [2, E]

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x=x, edge_index=edges)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.convs[-1](x, edges)
        return x



class Attention_Encoder(nn.Module):
    def __init__(self, heads, in_dim, out_dim, dropout):
        super(Attention_Encoder, self).__init__()
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.attention = nn.MultiheadAttention(in_dim, heads, dropout=dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.activation = nn.ReLU()

    def forward(self, Q, K, V, mask):
        res = Q
        Q_attn, _ = self.attention(Q, K, V, key_padding_mask=mask)
        Q_attn = self.norm(Q_attn + res)
        Q_attn = self.activation(Q_attn)

        res = Q_attn
        Q_ffn = self.linear(Q_attn)
        Q_ffn = self.norm(Q_ffn + res)

        return Q_ffn



class Sequential_model(nn.Module):
    def __init__(self, args):
        super(Sequential_model, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.encode_attn = Attention_Encoder(args.heads, args.emb_dim, args.emb_dim, args.dropout)
        self.decode_attn = nn.MultiheadAttention(args.emb_dim, args.heads, dropout=args.dropout)

    def forward(self, check_ins, poi_emb, check_in_length, training=True):
        check_ins = check_ins.permute(1, 0, 2)
        idx = torch.arange(check_ins.shape[0]).repeat(check_ins.shape[1], 1).to(check_ins.device)
        mask = torch.gt(idx.t(), check_in_length).t()
        # check_ins, weights = self.encode_attn(check_ins, check_ins, check_ins, key_padding_mask=mask)
        check_ins = self.encode_attn(check_ins, check_ins, check_ins, mask)

        interests, _ = self.decode_attn(poi_emb, check_ins, check_ins, key_padding_mask=mask)
        interests = F.dropout(interests, p=self.dropout, training=training).squeeze(0)
        # return similarity
        return interests


class HKGAT(nn.Module):
    def __init__(self, args, edges, relations):
        super(HKGAT, self).__init__()
        self.gat_model = args.gat_model
        self.HKG = HSimplE(args)

        self.GAT = CEGAT(in_dim=args.emb_dim,
                            hid_dim=args.hid_dim,  # Use args.enc_hidden to control the number of hidden layers
                            out_dim=args.emb_dim,
                            num_layers=args.num_layers,
                            heads=args.heads,
                            output_heads=args.output_heads,
                            dropout=args.dropout,
                            Normalization=args.normalization)

        self.sequential_model = Sequential_model(args)

        self.entity_num = args.entity_num

        self.E = nn.Embedding(args.entity_num + 1, args.emb_dim, padding_idx=0)
        self.R = nn.Embedding(args.relation_num, args.rel_emb_dim, padding_idx=0)

        self.check_in_encoder = nn.Sequential(
            nn.Linear(8 * args.emb_dim, args.emb_dim)
        )

        self.time_encoder_poi = nn.Sequential(
            nn.Linear(4 * args.emb_dim, args.emb_dim),
        )

        self.E.weight.data = torch.randn(args.entity_num + 1, args.emb_dim)
        self.R.weight.data = torch.randn(args.relation_num, args.rel_emb_dim)

        self.edges = edges
        self.relations = relations  # Contains all relations in the dataset
        self.poi_index = torch.range((args.index['poi']['start']), (args.index['poi']['end']), 1).long()

    def forward(self, index, mode, targets=None, length=None, ms=None, bs=None):
        if mode == 'kg':
            r_index = index[:, 0]
            e_index = index[:, 1:]
            r = self.R(r_index)
            E = self.E(e_index)

            return self.HKG(r, E, ms, bs)

        elif mode == 'train_gat_check_in' or mode == 'test_gat_check_in':
            # last_check_in = index[0, length[0] - 1
            labels = targets[:, 1]
            time1 = targets[:, 5]
            time2 = targets[:, 6]
            time3 = targets[:, 7]

            self.E.requires_grad_(False)

            if mode == 'train_gat_check_in':
                x = self.GAT(torch.cat((self.E.weight, self.E_GNN.weight)), self.edges, training=True)
            else:
                x = self.GAT(torch.cat((self.E.weight, self.E_GNN.weight)), self.edges, training=False)

            check_in_emb = torch.cat((x[index[:, :, 0]], x[index[:, :, 1]], x[index[:, :, 2]], x[index[:, :, 3]],
                                      x[index[:, :, 4]], x[index[:, :, 5]], x[index[:, :, 6]], x[index[:, :, 7]]),
                                     dim=2)
            
            check_in_emb = self.check_in_encoder(check_in_emb).squeeze()
            poi_emb = x[self.poi_index]

            poi_emb = torch.cat((poi_emb.reshape(poi_emb.shape[0], 1, -1).repeat(1, len(labels), 1),
                                 x[time1.reshape(1, -1).repeat(poi_emb.shape[0], 1)]
                                 , x[time2.reshape(1, -1).repeat(poi_emb.shape[0], 1)],
                                 x[time3.reshape(1, -1).repeat(poi_emb.shape[0], 1)]), dim=2)
            poi_emb = self.time_encoder_poi(poi_emb)

            if mode == 'train_gat_check_in':
                interests = self.sequential_model(check_in_emb, poi_emb, length, training=True)
            else:
                interests = self.sequential_model(check_in_emb, poi_emb, length, training=False)
            interests = interests.view(-1, 1, interests.shape[-1])
            poi_emb = poi_emb.view(-1, 1, poi_emb.shape[-1]).permute(0, 2, 1)
            similarity = torch.bmm(interests, poi_emb)
            similarity = similarity.view(-1, len(labels)).t()
            return similarity, labels
