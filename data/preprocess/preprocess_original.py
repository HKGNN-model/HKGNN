import collections
import random
import torch
import utils.sampler as sampler
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

class Dataset_Flashback(Dataset):

    def reset(self):
        # reset training state:
        self.next_user_idx = 0  # current user index to add
        self.active_users = []  # current active users
        self.active_user_seq = []  # current active users sequences
        self.user_permutation = []  # shuffle users during training

        # set active users:
        for i in range(self.args.fb_batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)

        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self):
        random.shuffle(self.user_permutation)
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.args.fb_batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)

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

        is_first = True
        user = 0
        current_user = 0
        for i, traj in enumerate(trajectories):
            if is_first is True:
                is_first = False
                current_user = traj[0][0]
            if traj[0][0] != current_user:
                user += 1
                current_user = traj[0][0]

            for j in range(0, length[i]):
                traj[j][0] = user
            trajectories[i] = traj

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
            label = []
            label_time = []
            label_coords = []

            for j in range(0, len(traj)):
                seq.append(traj[j][1])
                seq_time.append(traj[j][3])
                seq_coords.append(traj[j][2])
                label.append(targets[i][j][1])
                label_time.append(targets[i][j][3])
                label_coords.append(targets[i][j][2])
            self.sequences[user].append(seq)
            self.sequences_time[user].append(seq_time)
            self.sequences_coords[user].append(seq_coords)
            self.sequences_labels[user].append(label)
            self.sequences_lbl_times[user].append(label_time)
            self.sequences_lbl_coords[user].append(label_coords)
            self.sequences_lengths[user].append(length[i])
            self.sequences_count[user] += 1

        for user in self.sequences_count.keys():
            if self.sequences_count[user] > self.max_seq_count:
                self.max_seq_count = self.sequences_count[user]
            self.capacity += self.sequences_count[user]

        users = []
        sequences = []
        sequences_time = []
        sequences_coords = []
        sequences_labels = []
        sequences_lbl_times = []
        sequences_lbl_coords = []
        sequences_lengths = []

        for user in self.sequences.keys():
            users.append(user)
            sequences.append(self.sequences[user])
            sequences_time.append(self.sequences_time[user])
            sequences_coords.append(self.sequences_coords[user])
            sequences_labels.append(self.sequences_labels[user])
            sequences_lbl_times.append(self.sequences_lbl_times[user])
            sequences_lbl_coords.append(self.sequences_lbl_coords[user])
            sequences_lengths.append(self.sequences_lengths[user])
        #
        #
        self.users = users
        self.sequences = sequences
        self.sequences_times = sequences_time
        self.sequences_coords = sequences_coords
        self.sequences_labels = sequences_labels
        self.sequences_lbl_times = sequences_lbl_times
        self.sequences_lbl_coords = sequences_lbl_coords
        self.sequences_lengths = sequences_lengths

        self.reset()

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

        seqs = []
        times = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_coords = []
        reset_h = []
        lengths = []
        for i in range(self.args.fb_batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if (j >= max_j):
                # repalce this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)

            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]).long())
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            for k in range(len(self.sequences_coords[i_user][j])):
                if self.sequences_coords[i_user][j][k] == 0:
                    self.sequences_coords[i_user][j][k] = [0, 0]
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]).long())
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            lengths.append(torch.tensor(self.sequences_lengths[i_user][j]).long())
            self.active_user_seq[i] += 1

        x = torch.stack(seqs, dim=1)
        t = torch.stack(times, dim=1)
        s = torch.stack(coords, dim=1)
        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        lengths = torch.stack(lengths, dim=0)

        return x, t, s, y, y_t, y_s, reset_h, torch.tensor(self.active_users), lengths

    def __len__(self):
        ''' Amount of available batches to process each sequence at least once.
        '''

        estimated = self.capacity // self.args.fb_batch_size
        # return max(self.max_seq_count, estimated)
        return estimated



def get_dataset(check_in):
    train_trajectories, train_length, targets_train, val_trajectories, val_length, \
                                targets_val, test_trajectories, test_length, targets_test \
                                                        = sampler.extract_trajectories(check_in)


    check_in_count = collections.Counter(list(targets_train[:, 1]) + list(targets_test[:, 1]))
    # find 10% least popular pois

    check_in_count = sorted(check_in_count.items(), key=lambda x: x[1])
    check_in_count = check_in_count[:int(len(check_in_count) * 0.3)]
    check_in_least = set([v[0] for v in check_in_count])
    test_trajectories_least = [test_trajectories[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]
    test_length_least = [test_length[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]
    targets_test_least = [targets_test[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]

    val_trajectories_least = [val_trajectories[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]
    val_length_least = [val_length[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]
    targets_val_least = [targets_val[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]

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
                                targets_val, test_trajectories, test_length, targets_test \
                                                        = sampler.extract_consecutive_trajectories(check_ins)


    check_in_count = collections.Counter(list(targets_train[:, 1]) + list(targets_test[:, 1]))
    # find 10% least popular pois

    check_in_count = sorted(check_in_count.items(), key=lambda x: x[1])
    check_in_count = check_in_count[:int(len(check_in_count) * 0.3)]
    check_in_least = set([v[0] for v in check_in_count])
    test_trajectories_least = [test_trajectories[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]
    test_length_least = [test_length[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]
    targets_test_least = [targets_test[i] for i in range(len(test_trajectories)) if int(targets_test[i][1]) in check_in_least]

    val_trajectories_least = [val_trajectories[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]
    val_length_least = [val_length[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]
    targets_val_least = [targets_val[i] for i in range(len(val_trajectories)) if int(targets_val[i][1]) in check_in_least]

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









