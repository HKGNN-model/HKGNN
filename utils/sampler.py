import random
import numpy as np
import math
import torch
from torch.nn import functional as F


def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where(targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if (ind == len(positive_indices) - 1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq


def padd_and_decompose(targets, predictions, max_length):
    '''
    :param targets: tensor of shape (batch_size, max_length)
    :param predictions: tensor of shape (batch_size, max_length)
    :param max_length: maximum length of a sequence
    :return: tensor of shape (batch_size, max_length)
    '''
    seq = decompose_predictions(targets, predictions, max_length)
    return torch.stack(seq)


def neg_each(arr, arity, entity_num, neg_ratio):
    for a in range(arity):
        arr[a * neg_ratio + 1:(a + 1) * neg_ratio + 1, a + 1] = np.random.randint(low=1, high=entity_num,
                                                                                  size=neg_ratio)
    return arr


def padd(a, max_length):
    b = F.pad(a, (0, max_length - len(a)), 'constant', -math.inf)
    return b


def hkg_negative_sample(pos_batch, entity_num, neg_ratio, max_arity):
    '''
    :param pos_batch: list of positive samples
    :param entity_num: number of entities
    :param neg_ratio: number of negative samples for each positive sample
    :param max_arity: maximum arity of relations
    :return: negative samples, labels, arities
    '''
    relation_type = [relation[0] for relation in pos_batch]
    relation = [list(relation[1]) for relation in pos_batch]
    relation = [np.array([relation_type[i]] + relation[i] + [0]) for i in range(len(relation))]
    # pos_batch = np.append(relation, np.zeros((len(relation), 1)), axis=1).astype("int")
    arities = [len(r) - 2 for r in relation]

    neg_batch = []
    for i, c in enumerate(relation):
        c = np.array(list(c) + [entity_num] * (max_arity - arities[i]))
        neg_batch.append(
            neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], entity_num, neg_ratio))
    labels = []
    batch = []
    arities_new = []
    for i in range(len(neg_batch)):
        labels.append(1)
        labels = labels + [0] * (neg_ratio * arities[i])
        arities_new = arities_new + [arities[i]] * (neg_ratio * arities[i] + 1)

        for j in range(len(neg_batch[i])):
            batch.append(neg_batch[i][j][:-1])
    labels = np.array(labels)
    batch = np.array(batch)

    ms = np.zeros((len(batch), max_arity))
    bs = np.ones((len(batch), max_arity))
    for i in range(len(batch)):
        ms[i][0:arities_new[i]] = 1
        bs[i][0:arities_new[i]] = 0
    return batch, labels, ms, bs

def extract_train_trajectory(user_check_in, seq_length=100):
    trajectories = []
    trajectories_length = []
    targets = []
    for key in user_check_in:
        trajectory = None
        for i in range(0, len(user_check_in[key]) - 1):
            if trajectory is None:
                trajectory = user_check_in[key][i].reshape(1, -1)
                continue
            else:
                trajectory = np.vstack((trajectory, user_check_in[key][i]))
            if trajectory.shape[0] <= seq_length:
                targets.append(trajectory[-1])
                trajectories_length.append(trajectory.shape[0] - 1)
                pad_trajectory = np.vstack(
                    (trajectory[:-1], np.zeros((seq_length + 1 - trajectory.shape[0], trajectory.shape[1]))))
                trajectories.append(pad_trajectory)
            else:
                targets.append(trajectory[-1])
                trajectories_length.append(seq_length)
                trajectories.append(trajectory[:-1])
                trajectory = trajectory[-seq_length:]
    return trajectories, trajectories_length, targets

def extract_test_trajectory(user_check_in, user_length, seq_length=100, mode='test'):
    trajectories = []
    trajectories_length = []
    targets = []
    for key in user_check_in:
        trajectory = None
        for i in range(0, len(user_check_in[key])):
            if trajectory is None:
                trajectory = user_check_in[key][i].reshape(1, -1)
                continue
            else:
                trajectory = np.vstack((trajectory, user_check_in[key][i]))
            if (mode == 'test' and i >= user_length[key] * 0.8) or (mode == 'val' and user_length[key] * 0.8 > i >= user_length[key] * 0.7):
                if trajectory.shape[0] <= seq_length:
                    targets.append(trajectory[-1])
                    trajectories_length.append(trajectory.shape[0] - 1)
                    pad_trajectory = np.vstack((trajectory[:-1], np.zeros((seq_length + 1 - trajectory.shape[0], trajectory.shape[1]))))
                    trajectories.append(pad_trajectory)
                else:
                    targets.append(trajectory[-1])
                    trajectories_length.append(seq_length)
                    trajectories.append(trajectory[-seq_length - 1:-1])
    return trajectories, trajectories_length, targets

def extract_trajectories(check_in, seq_length=100):
    check_in = np.array(check_in)
    user_check_in = {}
    train_user_check_in = {}
    test_user_check_in = {}
    val_user_check_in = {}
    user_length = {}

    for i in range(0, len(check_in)):
        if check_in[i][0] not in user_check_in:
            user_check_in[check_in[i][0]] = []
        user_check_in[check_in[i][0]].append(check_in[i])

    for user in user_check_in:
        if len(user_check_in[user]) < 3:
            continue
        user_length[user] = len(user_check_in[user])
        train_num = int(len(user_check_in[user]) * 0.7)
        val_num = int(len(user_check_in[user]) * 0.1)
        train_user_check_in[user] = user_check_in[user][:train_num]
        val_user_check_in[user] = user_check_in[user][:train_num + val_num]
        test_user_check_in[user] = user_check_in[user]

    train_trajectories, train_trajectories_length, train_targets = extract_train_trajectory(train_user_check_in, seq_length=seq_length)
    train_trajectories = np.array(train_trajectories)
    train_trajectories_length = np.array(train_trajectories_length)
    train_targets = np.array(train_targets)


    val_trajectories, val_trajectories_length, val_targets = extract_test_trajectory(val_user_check_in, user_length,seq_length=seq_length, mode='val')
    val_trajectories = np.array(val_trajectories)
    val_trajectories_length = np.array(val_trajectories_length)
    val_targets = np.array(val_targets)

    test_trajectories, test_trajectories_length, test_targets = extract_test_trajectory(test_user_check_in, user_length,seq_length=seq_length, mode='test')
    test_trajectories = np.array(test_trajectories)
    test_trajectories_length = np.array(test_trajectories_length)
    test_targets = np.array(test_targets)

    return train_trajectories, train_trajectories_length, train_targets, val_trajectories, val_trajectories_length, \
           val_targets, test_trajectories, test_trajectories_length, test_targets


def kg_negtive_sample(args, pos_samples, all_triplets):
    entity_num = args.entity_num
    h, r, t = pos_samples[0], pos_samples[1], pos_samples[2]
    h_neg, t_neg = torch.zeros_like(h), torch.zeros_like(t)
    for i in range(len(h)):
        change_head = np.random.binomial(1, 0.5)
        if change_head:
            while True:
                h_neg[i] = np.random.randint(0, entity_num)
                if (h_neg[i], r[i], t[i]) not in all_triplets:
                    t_neg[i] = t[i]
                    break
        else:
            while True:
                t_neg[i] = np.random.randint(0, entity_num)
                if (h[i], r[i], t_neg[i]) not in all_triplets:
                    h_neg[i] = h[i]
                    break
    return (h_neg, t_neg)


    
