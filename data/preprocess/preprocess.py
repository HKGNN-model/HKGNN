import collections
import random
import time
from tqdm import tqdm

import numpy as np
import torch
import utils.sampler as sampler
import utils.util as util
from torch.utils.data.dataset import Dataset



class Dataset_HKG(Dataset):
    def __init__(self, trajectories, length, targets):
        self.trajectories = trajectories
        self.length = length
        self.targets = targets
    def __getitem__(self, idx):
        return self.trajectories[idx], self.length[idx], self.targets[idx]
    def __len__(self):
        return len(self.trajectories)
    
class Dataset_KG(Dataset):
    def __init__(self, triplets):
        triplets = torch.tensor(triplets, dtype=torch.long)
        self.h = triplets[:, 0]
        self.t = triplets[:, 1]
        self.r = triplets[:, 2]
    def __getitem__(self, idx):
        return self.h[idx], self.r[idx], self.t[idx]
    def __len__(self):
        return len(self.h)

class Dataset_Flashback(Dataset):
    def __init__(self, trajectories, length, targets, args):
        self.trajectories = trajectories
        self.length = length
        self.targets = targets
        self.args = args
        self.sequences = {}
        self.sequences_time = {}
        self.sequences_coords = {}
        self.sequences_labels = {}
        self.sequences_lbl_times = {}
        self.sequences_lbl_coords = {}
        self.sequences_lengths  = {}
        self.sequences_count = {}
        user_num = 0

        self.max_seq_count = 0
        self.capacity = 0

        for i, traj in enumerate(trajectories):

            user = traj[0][0]

            if user not in self.sequences:
                self.sequences[user] = []
                self.sequences_time[user] = []
                self.sequences_coords[user] = []
                self.sequences_labels[user] = []
                self.sequences_lbl_times[user] = []
                self.sequences_lbl_coords[user] = []
                self.sequences_lengths[user] = []
                self.sequences_count[user] = 0


            seq = []
            seq_time = []
            seq_coords = []
            for j in range(0, len(traj)):
                seq.append(traj[j][1])
                seq_time.append(traj[j][3])
                seq_coords.append(traj[j][2])
            self.sequences[user].append(seq)
            self.sequences_time[user].append(seq_time)
            self.sequences_coords[user].append(seq_coords)
            self.sequences_labels[user].append(targets[i][1])
            self.sequences_lbl_times[user].append(targets[i][3])
            self.sequences_lbl_coords[user].append(targets[i][2])
            self.sequences_lengths[user].append(length[i])
            self.sequences_count[user] += 1

        for user in self.sequences_count.keys():
            if self.sequences_count[user] > self.max_seq_count:
                self.max_seq_count = self.sequences_count[user]
            self.capacity += self.sequences_count[user]

        users = []
        sequences = []
        sequences_users = []
        sequences_time = []
        sequences_coords = []
        sequences_labels = []
        sequences_lbl_times = []
        sequences_lbl_coords = []
        sequences_lengths = []

        for user in self.sequences.keys():
            users.append(user)
            sequences_users = sequences_users + [user] * len(self.sequences[user])
            sequences = sequences + self.sequences[user]
            sequences_time = sequences_time + self.sequences_time[user]
            sequences_coords = sequences_coords + self.sequences_coords[user]
            sequences_labels = sequences_labels + self.sequences_labels[user]
            sequences_lbl_times = sequences_lbl_times + self.sequences_lbl_times[user]
            sequences_lbl_coords = sequences_lbl_coords + self.sequences_lbl_coords[user]
            sequences_lengths = sequences_lengths + self.sequences_lengths[user]

        for i in range(0, len(sequences_coords)):
            for j in range(0, len(sequences_coords[i])):
                if sequences_coords[i][j] == 0:
                    sequences_coords[i][j] = [0, 0]

        #
        #
        self.users = users
        self.sequences_users = np.array(sequences_users)
        self.sequences = np.array(sequences)
        self.sequences_time = np.array(sequences_time)
        self.sequences_coords = np.array(sequences_coords)
        self.sequences_labels = np.array(sequences_labels)
        self.sequences_lbl_times = np.array(sequences_lbl_times)
        self.sequences_lbl_coords = np.array(sequences_lbl_coords)
        self.sequences_lengths = np.array(sequences_lengths)


    def __getitem__(self, idx):
        ''' Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.

        y is the target location and y_t, y_s the targets timestamp and coordiantes. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset this users hidden state to initial value h_0.
        '''

        seqs = self.sequences[idx]
        times = self.sequences_time[idx]
        coords = list(self.sequences_coords[idx])
        lbls = self.sequences_labels[idx]
        lbl_times = self.sequences_lbl_times[idx]
        lbl_coords = list(self.sequences_lbl_coords[idx])
        lengths = self.sequences_lengths[idx]
        users = self.sequences_users[idx]

        x = torch.tensor(seqs, dtype=torch.long)
        t = torch.tensor(times, dtype=torch.long)
        s = torch.tensor(coords, dtype=torch.float)
        y = torch.tensor(lbls, dtype=torch.long)
        y_t = torch.tensor(lbl_times, dtype=torch.long)
        y_s = torch.tensor(lbl_coords, dtype=torch.float)
        lengths = torch.tensor(lengths, dtype=torch.long)
        # reset_h = torch.tensor(reset_h, dtype=torch.long)
        users = torch.tensor(users, dtype=torch.long)

        return x, t, s, y, y_t, y_s, users, lengths



    def __len__(self):
        ''' Amount of available batches to process each sequence at least once.
        '''

        return len(self.sequences)


class Dataset_Stan(Dataset):
    def __init__(self, trajectories, length, targets, args, mat2s, mode=None):
        self.args = args
        self.trajectories = torch.tensor(trajectories)
        self.length = length
        self.targets = targets
        self.mat1 = []
        self.mat2t = []
        device = torch.device('cuda:{}'.format(args.cuda) if args.use_gpu else 'cpu')
        self.mat2s = torch.tensor(mat2s, dtype=torch.float).to(device)
        mat1_max_1, mat1_min_1, mat1_max_2, mat1_min_2 = -9999, 9999, -9999, 9999
        if mode == 'train':
            for i in tqdm(range(len(self.trajectories)), desc='Processing trajectories'):
                mat_1 = util.rst_mat1(self.trajectories[i][:self.length[i]])
                mat1_max_1 = max(mat1_max_1, mat_1[:,:,0].max())
                mat1_min_1 = min(mat1_min_1, mat_1[:,:,0].min())
                mat1_max_2 = max(mat1_max_2, mat_1[:,:,1].max())
                mat1_min_2 = min(mat1_min_2, mat_1[:,:,1].min())
            self.ex = (mat1_max_1, mat1_min_1, mat1_max_2, mat1_min_2)

        #

        # self.mat1 = torch.stack(self.mat1)
        # self.mat2t = torch.stack(self.mat2t)
        # self.trajectories = self.trajectories[:, :, :3]
        # self.trajectories[:,:,1] = self.trajectories[:,:,1] - args.index['poi']['start']
        self.targets = self.targets[:, 1].astype(float) - args.index['poi']['start']

    def __getitem__(self, item):
        # trajectory = self.trajectories[item]
        trajectory = torch.tensor(self.trajectories[item], dtype=torch.long)
        length = torch.tensor(self.length[item], dtype=torch.long)
        target = torch.tensor(self.targets[item], dtype=torch.long)
        # mat1 = self.mat1[item]
        # mat2t = self.mat2t[item]
        # mat1 = torch.zeros((len(trajectory),self.args.stan_seq_len, self.args.stan_seq_len, 2)).float()
        # mat2t = torch.zeros((len(trajectory),self.args.stan_seq_len)).float()
        init_mat1 = torch.zeros((self.args.stan_seq_len, self.args.stan_seq_len, 2))
        init_mat2t = torch.zeros((self.args.stan_seq_len))
        real_trajectory = self.trajectories[item][:length.item()]
        mat_1 = util.rst_mat1(real_trajectory)
        mat_2t = util.rt_mat2t(real_trajectory[:, 2], length.item())
        init_mat1[:length.item(), :length.item()] = mat_1
        init_mat2t[:length.item()] = mat_2t
        # mat1[i] = init_mat1
        # mat2t[i] = init_mat2t
        trajectory = trajectory[:, :3]
        trajectory[:, 1] -= self.args.index['poi']['start']
        trajectory[:, 1][trajectory[:, 1] < 0] = 0

        return trajectory, length, target, init_mat1, init_mat2t

    def __len__(self):
        return len(self.trajectories)
    
class Dataset_Deepmove(Dataset):
    def __init__(self, trajectories, length, targets, history, args):
        self.args = args
        self.trajectories = torch.tensor(trajectories)
        self.length = length
        self.targets = targets
        self.history = history
        self.loc = []
        self.time = []
        self.loc_history = []
        self.time_history = []
        for i in range(len(self.trajectories)):
            self.loc.append(self.trajectories[i][:, 1])
            self.time.append(self.trajectories[i][:, 3])
            self.loc_history.append(self.history[i][:, 1])
            self.time_history.append(self.history[i][:, 3])
            
        self.loc = torch.tensor(self.loc)
        self.time = torch.tensor(self.time)
        self.loc_history = torch.tensor(self.loc_history)
        self.time_history = torch.tensor(self.time_history)

        self.loc -= args.index['poi']['start']
        self.loc[self.loc < 0] = 0 
        self.loc_history -= args.index['poi']['start']
        self.loc_history[self.loc_history < 0] = 0
        self.targets = self.targets[:, 1].astype(float) - args.index['poi']['start']

    def __getitem__(self, item):
        return self.loc[item], self.time[item], self.loc_history[item], self.time_history[item], self.length[item], self.targets[item]
    
    def __len__(self):
        return len(self.trajectories)

def get_dataset(check_in, seq_len=100):
    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test \
                                                        = sampler.extract_trajectories(check_in, seq_len)


    check_in_count = collections.Counter(list(targets_train[:, 1]) + list(targets_val[:, 1]) + list(targets_test[:, 1]))
    # find 10% least popular pois

    check_in_count = sorted(check_in_count.items(), key=lambda x: x[1])
    check_in_count = check_in_count[:int(len(check_in_count) * 0.3)]
    check_in_least = set([v[0] for v in check_in_count])
    test_trajectories_least = np.array([test_trajectories[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])
    test_length_least = np.array([test_length[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])
    targets_test_least = np.array([targets_test[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])

    val_trajectories_least = np.array([val_trajectories[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])
    val_length_least = np.array([val_length[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])
    targets_val_least = np.array([targets_val[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])

    print('The number of samples in the least popular pois: ', len(test_trajectories_least))
    print('The number of samples in the least alll pois: ', len(test_trajectories))

    return train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least

def get_dataset_deepmove(check_in, seq_len=100):
    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test \
                                                        = sampler.extract_trajectories_deepmove(check_in, seq_len)


    check_in_count = collections.Counter(list(targets_train[:, 1]))
    # find 10% least popular pois

    check_in_count = sorted(check_in_count.items(), key=lambda x: x[1])
    check_in_count = check_in_count[:int(len(check_in_count) * 0.3)]
    check_in_least = set([v[0] for v in check_in_count])
    test_trajectories_least = np.array([test_trajectories[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])
    test_length_least = np.array([test_length[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])
    targets_test_least = np.array([targets_test[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least])

    val_trajectories_least = np.array([val_trajectories[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])
    val_length_least = np.array([val_length[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])
    targets_val_least = np.array([targets_val[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least])

    return train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least

def preprocess_data_HKG(args, all_relations, train_data, poi_index):
    #######################################
    # prepare poi data
    check_ins = train_data['check_in']

    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least = get_dataset(check_ins)


    args.num_poi = poi_index['end'] - poi_index['start'] + 1
    args.index = {}
    args.index['poi'] = poi_index
    device = torch.device('cuda:{}'.format(args.cuda) if args.use_gpu else 'cpu')

    train_trajectories = torch.LongTensor(train_trajectories).to(device)
    train_length = torch.LongTensor(train_length).to(device)
    test_trajectories = torch.LongTensor(test_trajectories).to(device)
    test_length = torch.LongTensor(test_length).to(device)
    test_trajectories_least = torch.LongTensor(test_trajectories_least).to(device)
    test_length_least = torch.LongTensor(test_length_least).to(device)
    val_trajectories = torch.LongTensor(val_trajectories).to(device)
    val_length = torch.LongTensor(val_length).to(device)
    val_trajectories_least = torch.LongTensor(val_trajectories_least).to(device)
    val_length_least = torch.LongTensor(val_length_least).to(device)

    targets_train = torch.LongTensor(targets_train).to(device)
    targets_test = torch.LongTensor(targets_test).to(device)
    targets_test_least = torch.LongTensor(targets_test_least).to(device)
    targets_val = torch.LongTensor(targets_val).to(device)
    targets_val_least = torch.LongTensor(targets_val_least).to(device)

    ########################################
    # prepare knowledge graph data
    max_arity = max([len(relation[1]) for relation in all_relations])
    args.arity = max_arity
    #######################################

    train_data = {}
    train_data['KG'] =  all_relations
    train_data['GNN'] = {}
    train_data['GNN']['trajectories'] = train_trajectories
    train_data['GNN']['length'] = train_length
    train_data['GNN']['targets'] = targets_train

    valid_data = {}
    valid_data['all'] = {}
    valid_data['least'] = {}
    valid_data['all']['trajectories'] = val_trajectories
    valid_data['least']['trajectories'] = val_trajectories_least
    valid_data['all']['length'] = val_length
    valid_data['least']['length'] = val_length_least
    valid_data['all']['targets'] = targets_val
    valid_data['least']['targets'] = targets_val_least

    test_data = {}
    test_data['all'] = {}
    test_data['least'] = {}
    test_data['all']['trajectories'] = test_trajectories
    test_data['least']['trajectories'] = test_trajectories_least
    test_data['all']['length'] = test_length
    test_data['least']['length'] = test_length_least
    test_data['all']['targets'] = targets_test
    test_data['least']['targets'] = targets_test_least

    return args, train_data, valid_data, test_data


def preprocess_data_Flashback(args, check_ins, user_index, poi_index):

    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least = get_dataset(check_ins, seq_len=100)

    train_dataset = Dataset_Flashback(train_trajectories, train_length, targets_train, args)
    val_dataset = Dataset_Flashback(val_trajectories, val_length, targets_val, args)
    val_least_dataset = Dataset_Flashback(val_trajectories_least, val_length_least, targets_val_least, args)
    test_dataset = Dataset_Flashback(test_trajectories, test_length, targets_test, args)
    test_least_dataset = Dataset_Flashback(test_trajectories_least, test_length_least, targets_test_least, args)

    val_data = {}
    val_data['all'] = val_dataset
    val_data['least'] = val_least_dataset
    test_data = {}
    test_data['all'] = test_dataset
    test_data['least'] = test_least_dataset

    args.poi_num = poi_index['end'] - poi_index['start'] + 1
    args.user_num = user_index['end'] - user_index['start'] + 1
    args.index = {}
    args.index['poi'] = poi_index
    args.index['user'] = user_index

    return args, train_dataset, val_data, test_data

def preprocess_data_Graph_Flashback(args, check_ins, user_index, poi_index):

    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least = get_dataset(check_ins, seq_len=100)
    check_ins = np.array(check_ins)
    user_check_in = {}
    for i in range(0, len(check_ins)):
        if check_ins[i][0] not in user_check_in:
            user_check_in[check_ins[i][0]] = []
        user_check_in[check_ins[i][0]].append(check_ins[i])

    train_check_ins = []
    for user in user_check_in:
        if len(user_check_in[user]) < 3:
            continue
        train_num = int(len(user_check_in[user]) * 0.7)
        train_check_ins.append(user_check_in[user][:train_num])

    train_check_ins = np.concatenate(train_check_ins, axis=0)

    train_dataset = Dataset_Flashback(train_trajectories, train_length, targets_train, args)
    val_dataset = Dataset_Flashback(val_trajectories, val_length, targets_val, args)
    val_least_dataset = Dataset_Flashback(val_trajectories_least, val_length_least, targets_val_least, args)
    test_dataset = Dataset_Flashback(test_trajectories, test_length, targets_test, args)
    test_least_dataset = Dataset_Flashback(test_trajectories_least, test_length_least, targets_test_least, args)
    
    train_data = {}
    train_data['KG'] = train_check_ins
    train_data['Flashback'] = train_dataset

    val_data = {}
    val_data['all'] = val_dataset
    val_data['least'] = val_least_dataset
    test_data = {}
    test_data['all'] = test_dataset
    test_data['least'] = test_least_dataset

    args.poi_num = poi_index['end'] - poi_index['start'] + 1
    args.user_num = user_index['end'] - user_index['start'] + 1
    args.entity_num = args.poi_num + args.user_num
    args.index = {}
    args.index['poi'] = poi_index
    args.index['user'] = user_index

    return args, train_data, val_data, test_data



def process_data_Stan(args, check_ins, user_index, poi_index, poi_data):

    for index, check_in in check_ins.iterrows():
        check_in_time = check_in['time']
        check_in_time = time.localtime(check_in_time)
        year = check_in_time.tm_year - 2012
        month = check_in_time.tm_mon - 1
        day = check_in_time.tm_mday - 1
        node = year * 365 * 24 + month * 31 * 24 + day * 24 + check_in_time.tm_hour
        check_ins.loc[index, 'time'] = node

    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test, \
                                test_trajectories_least, test_length_least, targets_test_least, \
                                val_trajectories_least, val_length_least, targets_val_least = get_dataset(check_ins, seq_len=100)

    args.poi_num = poi_index['end'] - poi_index['start'] + 1
    args.user_num = user_index['end'] - user_index['start'] + 1
    args.index = {}
    args.index['poi'] = poi_index
    args.index['user'] = user_index

    mat2s = util.rs_mat2s(poi_data, len(poi_data))
    # mat2s = None
    train_dataset = Dataset_Stan(train_trajectories, train_length, targets_train, args, mat2s, mode='train')
    val_dataset = Dataset_Stan(val_trajectories, val_length, targets_val, args, mat2s)
    val_least_dataset = Dataset_Stan(val_trajectories_least, val_length_least, targets_val_least, args, mat2s)
    test_dataset = Dataset_Stan(test_trajectories, test_length, targets_test, args, mat2s)
    test_least_dataset = Dataset_Stan(test_trajectories_least, test_length_least, targets_test_least, args, mat2s)

    # ex = train_dataset.mat1[:, :, :, 0].max(), train_dataset.mat1[:, :, :, 0].min(),\
    #      train_dataset.mat1[:, :, :, 1].max(), train_dataset.mat1[:, :, :, 1].min()
    args.ex = train_dataset.ex
    val_data = {}
    val_data['all'] = val_dataset
    val_data['least'] = val_least_dataset
    test_data = {}
    test_data['all'] = test_dataset
    test_data['least'] = test_least_dataset



    return args, train_dataset, val_data, test_data






