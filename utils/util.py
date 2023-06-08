import random
import collections
import torch
import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp

from models.mymodel.model import HKGAT
from models.baselines.Flashback_model import Flashback,RnnFactory, GraphFlasback, TransEModel
from models.baselines.RNN_model import RNN
from models.baselines.STAN_model import STAN
import utils.sampler as sampler

#############################################################################################
'''
STAN util
'''
global_seed = 2023

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(np.sqrt(a))
    r = 6371
    return c * r


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1_rad = torch.deg2rad(lon1)
    lat1_rad = torch.deg2rad(lat1)
    lon2_rad = torch.deg2rad(lon2)
    lat2_rad = torch.deg2rad(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371.0
    return c * r

def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def rst_mat1(traj):
    poi = traj[..., [3,4]] 

    poi_item = poi.unsqueeze(1)  # shape: (*M, 1, [l, lat, lon])
    poi_term = poi.unsqueeze(0)  # shape: (1, *M, [l, lat, lon])

    lon1, lat1 = poi_item[..., 0], poi_item[..., 1]  # shape: (*M, 1)
    lon2, lat2 = poi_term[..., 0], poi_term[..., 1]  # shape: (1, *M)
    mat_dis = haversine(lon1, lat1, lon2, lat2)  # shape: (*M, *M)

    tim_item = traj[..., 2].unsqueeze(1)  # shape: (*M, 1)
    tim_term = traj[..., 2].unsqueeze(0)  # shape: (1, *M)
    mat_tim = torch.abs(tim_item - tim_term)  # shape: (*M, *M)

    mat = torch.stack([mat_dis, mat_tim], dim=-1)  # shape: (*M, *M, 2)

    return mat

def rs_mat2s(poi, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(0, l_max - 1, l_max)
    mat = np.zeros((l_max, l_max))  # mat (L, L)
    for i, loc1 in enumerate(candidate_loc):
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1)], poi[int(loc2)]  # retrieve poi by loc_id
            mat[i, j] = haversine_np(lon1=poi1[0], lat1=poi1[1], lon2=poi2[0], lat2=poi2[1])
    return mat  # (L, L)


def rt_mat2t(traj_time, length):
    item = traj_time[length - 1]
    mat = torch.abs(traj_time[:length] -item) 
    return mat


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)
#############################################################################################

#############################################################################################
'''
Graph Flashback utils
'''
def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W


def calculate_reverse_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    return calculate_random_walk_matrix(np.transpose(adj_mx))





#################################################################################################


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def get_model(args, edges=None, all_relations=None, mode=None):
    if args.model == 'HKGAT':
        model = HKGAT(args, edges, all_relations)
    elif args.model == 'Flashback':
        f_t = lambda delta_t: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*0.1)) # hover cosine + exp decay
        f_s = lambda delta_s: torch.exp(-(delta_s * 100)) # exp decay
        rnn_factory = RnnFactory(args.fb_rnn_type)
        model = Flashback(input_size=args.poi_num,user_count=args.user_num,hidden_size=args.fb_hidden_size,f_t=f_t,f_s=f_s,
                          rnn_factory=rnn_factory)
    elif args.model == 'GraphFlashback':
        if mode == 'kg':
            model = TransEModel(args.gfb_L1_flag, args.gfb_entity_dim, args.entity_num, 4)
        else:
            f_t = lambda delta_t: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*0.1)) # hover cosine + exp decay
            f_s = lambda delta_s: torch.exp(-(delta_s * 100)) # exp decay
            rnn_factory = RnnFactory('rnn')
            model = GraphFlasback(f_t=f_t,f_s=f_s,rnn_factory=rnn_factory,args=args)

    elif args.model == 'STAN':
        device = torch.device('cuda:{}'.format(args.cuda) if args.use_gpu else 'cpu')
        model = STAN(t_dim=args.t_dim, u_dim=args.user_num, l_dim=args.poi_num, embed_dim=args.stan_hidden_size, ex=args.ex, device=device, dropout=args.stan_dropout)
    else:
        raise NotImplementedError
    return model

















