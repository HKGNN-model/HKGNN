import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import argparse
import os

import utils.util as utils
import data.preprocess.data_processor as data_processor
import data.preprocess.preprocess as preprocess
from trainer import Trainer_HKG, Trainer_Flashback, Trainer_Stan, Trainer_Graph_Flashback
import utils.flashback_graph as flashback_graph


def train_hkgat(args):
    reload = False
    # load the data
    try:
        if reload:
            raise Exception
        with open('data/processed/{}/{}_data.pkl'.format(args.city, args.city), 'rb') as f:
            all_relations, relation_count, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num = pickle.load(f)
    except:
        # save processed data into pickle file
        all_relations, relation_count, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num = data_processor.process_data_HKG(args)
        if not os.path.exists('data/processed/{}'.format(args.city)):
            os.makedirs('data/processed/{}'.format(args.model))
        with open('data/processed/{}/{}_data.pkl'.format(args.city, args.city), 'wb') as f:
            pickle.dump([all_relations, relation_count, edge_index, test_data, train_data, user_index, poi_index, hyperedge_num, hypernode_num], f)

    args, train_data, val_data, test_data = preprocess.preprocess_data_HKG(args,all_relations, train_data, poi_index)

    device = torch.device('cuda:{}'.format(args.cuda) if args.use_gpu else 'cpu')

    # expand the hypergraph into a graph
    try:
        if reload:
            raise Exception
        with open('data/processed/{}/{}_expanded_edges.pkl'.format(args.city, args.city), 'rb') as f:
            expanded_edges, edge_weight = pickle.load(f)
    except:
        expanded_edges, edge_weight = data_processor.ConstructV2V(edge_index)
        expanded_edges = torch.LongTensor(expanded_edges)
        edge_weight = torch.FloatTensor(edge_weight)
        expanded_edges, edge_weight = data_processor.norm_contruction(expanded_edges, edge_weight)
        with open('data/processed/{}/{}_expanded_edges.pkl'.format(args.city, args.city), 'wb') as f:
            pickle.dump([expanded_edges, edge_weight], f)
    train_data['GNN']['edges'] = expanded_edges.to(device)

    args.relation_num = max(relation[0] for relation in all_relations) + 1
    args.entity_num = max([max(relation[1]) for relation in all_relations])
    args.hyperedge_num = hyperedge_num


    trainer = Trainer_HKG(args, train_data, val_data, test_data)
    trainer.train()
    return


def train_flashback(args):
    try:
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'rb') as f:
            check_ins, user_index, poi_index = pickle.load(f)
    except:
        # save processed data into pickle file
        check_ins, user_index, poi_index = data_processor.process_data_Flashback(args)
        if not os.path.exists('data/processed/baselines/{}'.format(args.model)):
            os.makedirs('data/processed/baselines/{}'.format(args.model))        
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'wb') as f:
            pickle.dump([check_ins, user_index, poi_index], f)
    args, train_data, val_data, test_data = preprocess.preprocess_data_Flashback(args, check_ins, user_index, poi_index)
    trainer = Trainer_Flashback(args, train_data, val_data, test_data)
    trainer.train()
    return


def train_graphflashback(args):
    try:
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'rb') as f:
            check_ins, user_index, poi_index, poi_data, friend_list = pickle.load(f)
    except:
        # save processed data into pickle file
        check_ins, user_index, poi_index, poi_data, friend_list = data_processor.process_data_Graph_Flashback(args)
        if not os.path.exists('data/processed/baselines/{}'.format(args.model)):
            os.makedirs('data/processed/baselines/{}'.format(args.model))        
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'wb') as f:
            pickle.dump([check_ins, user_index, poi_index, poi_data, friend_list], f)
    args, train_data, val_data, test_data = preprocess.preprocess_data_Graph_Flashback(args, check_ins, user_index, poi_index)
    # train KG data 
    kg_train_data = flashback_graph.generate_train_test_triplets(args, train_data['KG'], poi_data, friendship=friend_list)
    train_data['KG'] = kg_train_data
    trainer = Trainer_Graph_Flashback(args, train_data, val_data, test_data)
    trainer.train()
    return

def train_stan(args):
    try:
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'rb') as f:
            args, train_data, val_data, test_data= pickle.load(f)
    except:
        check_ins, user_index, poi_index, poi_data = data_processor.process_data_Stan(args)
        args, train_data, val_data, test_data = preprocess.process_data_Stan(args, check_ins, user_index, poi_index, poi_data)
        if not os.path.exists('data/processed/baselines/{}'.format(args.model)):
            os.makedirs('data/processed/baselines/{}'.format(args.model))
        with open('data/processed/baselines/{}/{}_data.pkl'.format(args.model, args.city), 'wb') as f:
            pickle.dump([args, train_data, val_data, test_data], f)
        
    trainer = Trainer_Stan(args, train_data, val_data, test_data)
    trainer.train()
    return




