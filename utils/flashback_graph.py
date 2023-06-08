'''
Implemented by https://github.com/kevin-xuan/Graph-Flashback
'''

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pandas as pd 
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import argparse

import utils.util as util
import data.preprocess.preprocess as preprocess

def calculate_score(h_e, t_e, rel, L1_flag=True, norm=None, proj=None):
    if L1_flag:
        score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e), 1))
    else:
        score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e) ** 2, 1))

    return score


def construct_user_poi_graph(args, user_encoder, loc_encoder, temporal_preference, norm=None, proj=None):
    
    loc_count = args.poi_num
    user_count = args.user_num

    threshold = args.gfb_threshold
    L1_flag = args.gfb_L1_flag

    bar = tqdm(total=user_count)
    bar.set_description('Construct User-POI Graph')

    transition_graph = lil_matrix((user_count, loc_count), dtype=np.float32)  # 有向图
    for i in range(user_count):
        h_e = user_encoder(torch.LongTensor([i]).to(args.device))
        t_list = list(range(loc_count))
        t_e = loc_encoder(torch.LongTensor(t_list).to(args.device))

        transition_vector = calculate_score(h_e, t_e, temporal_preference, L1_flag, norm, proj)
        indices = torch.argsort(transition_vector, descending=True)[:threshold]  # 选top_k
        norm = torch.max(transition_vector[indices])
        for index in indices:
            index = index.item()
            transition_graph[i, index] = (transition_vector[index] / norm).item()

        bar.update(1)
    bar.close()

    transition_graph = csr_matrix(transition_graph)

    return transition_graph

def construct_poi_poi_graph(args, loc_encoder, temporal_preference, norm=None, proj=None):
    loc_count = args.poi_num

    threshold = args.gfb_threshold
    L1_flag = args.gfb_L1_flag

    bar = tqdm(total=loc_count)
    bar.set_description('Construct Transition Graph')

    transition_graph = lil_matrix((loc_count, loc_count), dtype=np.float32)  # 有向图
    for i in range(loc_count):
        h_e = loc_encoder(torch.LongTensor([i]).to(args.device))
        t_list = list(range(loc_count))

        t_e = loc_encoder(torch.LongTensor(t_list).to(args.device))
        indices = torch.LongTensor(t_list[:i] + t_list[i + 1:]).to(args.device)

        transition_vector = calculate_score(h_e, t_e, temporal_preference, L1_flag, norm, proj)
        transition_vector_a = torch.index_select(transition_vector, 0, indices)  # [0, 1, ..., i-1, i+1, i+2, ...]

        indices = torch.argsort(transition_vector_a, descending=True)[:threshold]  # 选top_k
        norm = torch.max(transition_vector_a[indices])
        for index in indices:
            index = index.item()
            if index < i:
                pass
            else:
                index += 1
            transition_graph[i, index] = (transition_vector[index] / norm).item()

        bar.update(1)
    bar.close()

    transition_graph = coo_matrix(transition_graph)

    return transition_graph


def construct_friend_graph(args, user_encoder, friend_preference, norm=None, proj=None, friend_flag=True):
    loc_count = args.poi_num
    user_count = args.user_num

    threshold = args.gfb_threshold
    L1_flag = args.gfb_L1_flag


    bar = tqdm(total=user_count)
    if friend_flag:
        bar.set_description('Construct User friend Graph')
    else:
        bar.set_description('Construct User interact Graph')

    friend_graph = lil_matrix((user_count, user_count), dtype=np.float32)  # 有向图
    for i in range(user_count):
        if friend_flag:
            h_e = user_encoder(torch.LongTensor([i]).to(args.device))
            t_list = list(range(user_count))
            t_e = user_encoder(torch.LongTensor(t_list).to(args.device))
            indices = torch.LongTensor(t_list[:i] + t_list[i + 1:]).to(args.device)
            transition_vector = calculate_score(h_e, t_e, friend_preference, L1_flag, norm, proj)
            transition_vector_a = torch.index_select(transition_vector, 0, indices)  # [0, 1, ..., i-1, i+1, i+2, ...]
            indices = torch.argsort(transition_vector_a, descending=True)[:threshold]  # 选top_k
            norm = torch.max(transition_vector_a[indices])
            for index in indices:
                index = index.item()
                if index < i:
                    pass
                else:
                    index += 1
                friend_graph[i, index] = (transition_vector[index] / norm).item()
        else:
            h_e = user_encoder(torch.LongTensor([i]).to(args.device))
        bar.update(1)
    bar.close()
    friend_graph = coo_matrix(friend_graph)

    return friend_graph


def generate_train_test_triplets(args, check_ins, poi_data, friendship):  # 构造train/test 三元组
    triplets = []
    print('Construct interact relation and temporal relation......')
    users_count = args.user_num
    user_check_ins = {}

    for check_in in check_ins:
        user = check_in[0]
        poi = check_in[1]
        if user not in user_check_ins:
            user_check_ins[user] = []
        user_check_ins[user].append(poi)

    for user in user_check_ins.keys():
        # 构建interact关系
        for poi in user_check_ins[user]:
            triplets.append([user, poi, 0])  # 0代表interact relation

        # 构建temporal关系  相邻poi相连
        # print('Construct temporal relation......')
        for i in range(len(user_check_ins[user]) - 1):
            poi_prev = int(user_check_ins[user][i])
            poi_next = int(user_check_ins[user][i + 1])
            if poi_prev != poi_next:
                triplets.append([poi_prev, poi_next, 1])  # 1代表temporal relation

    # 构建spatial关系  两个poi的距离小于距离阈值lambda_d，就相连
    print('Construct spatial relation......')

    # 方案2
    lambda_d = 3  # 距离阈值为3千米, 再取top k, 即双重限制
    with tqdm(total=len(poi_data)) as bar:
        for i in range(len(poi_data)):
            loci_list = []
            for j in range(len(poi_data)):
                coord_prev = poi_data[i]
                coord_next = poi_data[j]
                lat_prev, lon_prev = coord_prev
                lat_next, lon_next = coord_next

                dist = util.haversine_np(lat_prev, lon_prev, lat_next, lon_next)
                if dist <= lambda_d :
                    loci_list.append((j, dist))  # 先是第一重限制, 这样可能会造成很多重复计算

            sort_list = sorted(loci_list, key=lambda x: x[1])  # 从小到大排序,距离越小,排名越靠前
            length = min(len(sort_list), 50)
            select_pois = sort_list[:length]  # 一般情况下, sort_list的长度肯定不止50, 取top 50  这是第二重限制
            for poi_entity, _ in select_pois:
                triplets.append([i, poi_entity, 2])
                
            bar.update(1)

    # 构建friend关系  互为朋友的user相连  这个train/test会重复构造一次,可以选择生成一个friend_triplet文件,然后再将其内容放入train/test
    # 但因为数量很少,构造很快,所以放在一起
    print('Construct friend relation......')
    # with open(friendship_file, 'r') as f_friend:
    #     for friend_line in f_friend.readlines():
    #         tokens = friend_line.strip('\n').split('\t')
    for i in range(len(friendship)):
        user_id1 = friendship[i][0]
        user_id2 = friendship[i][1]
        triplets.append([user_id1, user_id2, 3])
        triplets.append([user_id2, user_id1, 3])

    
    kg_dataset = preprocess.Dataset_KG(triplets)
    return kg_dataset
    