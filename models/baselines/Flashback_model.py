'''
Implementation of Flashback and GraphFlashback models
Mostly are from https://github.com/eXascaleInfolab/Flashback_code and https://github.com/kevin-xuan/Graph-Flashback
Changes are made to adapt to our training and testing process
''' 



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix

import utils.util as util

class Rnn(Enum):
    ''' The available RNN units '''

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    ''' Creates the desired RNN unit. '''

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class Flashback(nn.Module):
    ''' Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    '''

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)  # create outputs in lenght of locations


    def forward(self, x, t, s, y_t, y_s, h, active_user, length):
        seq_len, batch_size = x.size()
        x_emb = self.encoder(x)
        # print(self.encoder.weight)
        x_emb_packed = nn.utils.rnn.pack_padded_sequence(x_emb, length, enforce_sorted=False)
        out_packed, h = self.rnn(x_emb_packed, h)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, total_length=seq_len)
        out = out.permute(1, 0, 2)  # batch first
        # comopute weights per user
        out_w = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for i in range(batch_size):
            # sum_w = torch.zeros(1, device=x.device)
            dist_t = t[length[i] - 1, i].repeat(1, length[i]) - t[:length[i], i]
            dist_s = torch.norm(s[length[i] - 1, i].repeat(length[i], 1) - s[:length[i], i], dim=-1)
            a_j = self.f_t(dist_t).squeeze()
            b_j = self.f_s(dist_s)
            w_j = a_j * b_j + 1e-10# small epsilon to avoid 0 division
            sum_w = w_j.sum()
            out_w[i] = torch.sum(w_j.squeeze(0).unsqueeze(-1) * out[i, :length[i]], dim=0)
            out_w[i] /= sum_w

        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(batch_size, self.hidden_size)
        out_pu = torch.cat([out_w, p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear


'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h, c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h, c)
    

###############################################################
'''
GraphFlasback model below
'''


class TransEModel(nn.Module):
    def __init__(self,
                 L1_flag,
                 embedding_size,
                 ent_total,
                 rel_total
                 ):
        super(TransEModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.is_pretrained = False

        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        # init user and item embeddings
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_ent_emb
        self.rel_embeddings.weight.data = normalize_rel_emb


    def forward(self, h, r, t):
        h_e = self.ent_embeddings(h)
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)

        # L1 distance
        if self.L1_flag:
            score = torch.sum(torch.abs(h_e + r_e - t_e), 1)
        # L2 distance
        else:
            score = torch.sum((h_e + r_e - t_e) ** 2, 1)
        return score
    

class GraphFlasback(nn.Module):
    def __init__(self, f_t, f_s, rnn_factory, args):
        super().__init__()
        self.input_size = args.poi_num  
        self.user_count = args.user_num
        self.hidden_size = args.gfb_hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        self.I = identity(args.transition_graph.shape[0], format='coo')
        self.transition_graph = util.sparse_matrix_to_tensor(
            util.calculate_random_walk_matrix((args.transition_graph + self.I).astype(np.float32))).to(args.device)

        self.spatial_graph = args.spatial_graph
        if self.use_spatial_graph:
            self.spatial_graph = (self.spatial_graph + self.I).astype(np.float32)
            self.spatial_graph = util.calculate_random_walk_matrix(args.spatial_graph)
            self.spatial_graph = util.sparse_matrix_to_tensor(args.spatial_graph).to(args.device) # sparse tensor gpu
        self.friend_graph = args.friend_graph  # (M, N)
        if self.use_graph_user:
            I_f = identity(self.friend_graph.shape[0], format='coo')
            self.friend_graph = (self.friend_graph + I_f).astype(np.float32)
            self.friend_graph = util.calculate_random_walk_matrix(self.friend_graph)
            self.friend_graph = util.sparse_matrix_to_tensor(self.friend_graph).to(args.device)

        if args.interact_graph is not None:
            self.interact_graph = util.sparse_matrix_to_tensor(util.calculate_random_walk_matrix(args.interact_graph)).to(args.device)  # (M, N)
        else:
            self.interact_graph = None

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(self.user_count, self.hidden_size)  # user embedding
        self.rnn = rnn_factory.create(self.hidden_size) 
        self.fc = nn.Linear(2 * self.hidden_size, self.input_size)

    def forward(self, x, t, s, y_t, y_s, h, active_user, length):
        seq_len, batch_size = x.size()
        x_emb = self.encoder(x)

        if self.use_graph_user:
            friend_graph = self.friend_graph
            # AX
            user_emb = self.user_encoder(torch.LongTensor(list(range(self.user_count))).to(x.device))
            user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(x.device)  # (user_count, hidden_size)

            p_u = torch.index_select(user_encoder_weight, 0, active_user.squeeze())
        else:
            p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            # (user_len, hidden_size)
            p_u = p_u.view(batch_size, self.hidden_size)

        transition_graph = self.transition_graph
        loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(transition_graph, loc_emb) # (input_size, hidden_size)
        
        if self.use_spatial_graph:
            encoder_weight += torch.sparse.mm(self.spatial_graph,loc_emb)
            encoder_weight /= 2 
       
        new_x_emb = []
        for i in range(seq_len):
            # (user_len, hidden_size)
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)

        x_emb = torch.stack(new_x_emb, dim=0)

        # user-poi
        loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(x.device))
        encoder_weight = loc_emb
        interact_graph = self.interact_graph
        encoder_weight_user = torch.sparse.mm(interact_graph, encoder_weight)

        user_preference = torch.index_select(encoder_weight_user, 0, active_user.squeeze()).unsqueeze(0)

        user_loc_similarity = torch.exp(-(torch.norm(user_preference - x_emb, p=2, dim=-1)))
        user_loc_similarity = user_loc_similarity.permute(1, 0)

        x_emb_packed = nn.utils.rnn.pack_padded_sequence(x_emb, length, enforce_sorted=False)
        out_packed, h = self.rnn(x_emb_packed, h)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, total_length=seq_len)
        out = out.permute(1, 0, 2)  # batch first
        # comopute weights per user
        out_w = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for i in range(batch_size):

            dist_t = t[length[i] - 1, i].repeat(1, length[i]) - t[:length[i], i]
            dist_s = torch.norm(s[length[i] - 1, i].repeat(length[i], 1) - s[:length[i], i], dim=-1)
            a = self.f_t(dist_t).squeeze()
            b = self.f_s(dist_s)
            w = a * b * user_loc_similarity[i, :length[i]] # small epsilon to avoid 0 division
            sum_w = w.sum()
            out_w[i] = torch.sum(w.squeeze(0).unsqueeze(-1) * out[i, :length[i]], dim=0)
            out_w[i] /= sum_w

        # add user embedding:
        out_pu = torch.cat([out_w, p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear


