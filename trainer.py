import collections
import datetime
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torch.nn import Parameter
import argparse
import sklearn.metrics
import pickle
from torch.utils.tensorboard import SummaryWriter

import utils.sampler as sampler
import utils.metrics as metrics
import utils.util as utils
import utils.flashback_graph as flashback_graph
from data.preprocess.preprocess import Dataset_HKG

from models.baselines.Flashback_model import *

time_start = time.time()


class Trainer():
    def __init__(self, args):
        self.model_name = args.model

        self.args = args
        self.index = args.index
        log_dir = 'log/{}/{}_{}/'.format(args.city, args.model, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.save_dir = 'saved_model/{}_{}_{}_{}/'.format(args.city, args.model, args.use_kg, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_mrr = 0
        self.device = torch.device('cuda:{}'.format(args.cuda) if args.use_gpu else 'cpu')
        self.runs = args.runs

    def train(self):
        pass


    def loss(self, scores, labels):
        pass

    def get_model_prediction(self, inputs, mode='train') -> tuple:
        pass

    def cal_train_metric(self, total_iter, iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank, predictions, labels, loss, iteration, epoch, run):
            total_loss += loss.item()
            acc_1 += metrics.ACCK(predictions, labels, 1)
            acc_5 += metrics.ACCK(predictions, labels, 5)
            acc_10 += metrics.ACCK(predictions, labels, 10)
            acc_20 += metrics.ACCK(predictions, labels, 20)
            mrr += metrics.MRR(predictions, labels)
            avg_rank += metrics.avgrank(predictions, labels)
            iter += 1

            if iteration % self.args.show_iter == 0:
                global time_start
                time_end = time.time()
                time_consume = time_end - time_start
                total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = total_loss /iter, acc_1 / iter, acc_5 / iter, acc_10 / iter, acc_20 / iter, mrr / iter, avg_rank / iter
                print('epoch: {}, iteration: {}/{}, loss: {:.4f}, acc_1: {:.4f}, acc_5: {:.4f}, acc_10: {:.4f}, '
                      'acc_20: {:.4f}, mrr: {:.4f}, avg_rank: {:.4f}'.format(epoch, iteration, total_iter, total_loss ,
                                                                                acc_1, acc_5, acc_10, acc_20, mrr, avg_rank))
                print('Training for {} iterations consume {} s'.format(self.args.show_iter, time_consume))
                metrics_name = ['loss', 'acc_1', 'acc_5', 'acc_10', 'acc_20', 'mrr', 'avg_rank']
                for metric in metrics_name:
                    self.writer.add_scalar('Runs{}/GNN/{}'.format(run, metric), vars()[metric], total_iter)
                total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                iter = 0
                time_start = time.time()
            return iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank

    def evaluate(self, data_loader, dataset, epoch, run, mode='validate'):
        acc_1, acc_5, acc_10, acc_20, mrr, avg_rank, loss = 0, 0, 0, 0, 0, 0, 0
        iter = 0
        args = self.args

        with torch.no_grad():
            for i, inputs in enumerate(data_loader):

                predictions, labels = self.get_model_prediction(inputs, mode='test')

                iter += 1
                acc_1 += metrics.ACCK(predictions, labels, 1)
                acc_5 += metrics.ACCK(predictions, labels, 5)
                acc_10 += metrics.ACCK(predictions, labels, 10)
                acc_20 += metrics.ACCK(predictions, labels, 20)
                mrr += metrics.MRR(predictions, labels)
                avg_rank += metrics.avgrank(predictions, labels)
                loss += F.cross_entropy(predictions, labels)

            acc_1, acc_5, acc_10, acc_20, mrr, avg_rank, loss = acc_1 / iter, acc_5 / iter, acc_10 / iter, acc_20 / iter, mrr / iter, avg_rank / iter, loss / iter

            print('Evaluate at epoch {:d} on {} {} data'.format(epoch, dataset, mode),
                  'loss: {:.4f}'.format(loss),
                  'Acc@1: {:.4f}'.format(acc_1),
                  'Acc@5: {:.4f}'.format(acc_5),
                  'Acc@10: {:.4f}'.format(acc_10),
                  'Acc@20: {:.4f}'.format(acc_20),
                  'MRR: {:.4f}'.format(mrr),
                  'avg_rank: {:.4f}'.format(avg_rank))
            metrics_name = ['loss', 'acc_1', 'acc_5', 'acc_10', 'acc_20', 'mrr', 'avg_rank']
            for metric in metrics_name:
                self.writer.add_scalar('Runs: {}/{}/{}/{}'.format(run,mode, dataset, metric), vars()[metric], epoch)

    def save_model(self, epoch):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.model.state_dict(), self.save_dir + '{}_{}.pkl'.format(self.model_name, epoch))

    def evaluation(self,dataloader_all, dataloader_least, epoch, run, mode='validate'):

        print('Evaluate on all validation dataset')
        self.evaluate(dataloader_all, 'all', epoch, run, mode)

        print('Evaluate on least visited validation dataset')
        self.evaluate(dataloader_least, 'least', epoch, run, mode)


class Trainer_HKG(Trainer):
    def __init__(self, args, train_data, valid_data, test_data):
        super(Trainer_HKG, self).__init__(args)
        self.train_data_KG = train_data['KG']
        self.train_data_GNN = train_data['GNN']

        self.train_dataset = Dataset_HKG(train_data['GNN']['trajectories'], train_data['GNN']['length'], train_data['GNN']['targets'])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.gat_batch_size, shuffle=True, num_workers=0)

        self.valid_dataset_all = Dataset_HKG(valid_data['all']['trajectories'], valid_data['all']['length'], valid_data['all']['targets'])
        self.valid_dataset_least = Dataset_HKG(valid_data['least']['trajectories'], valid_data['least']['length'], valid_data['least']['targets'])
        self.valid_dataloader_all = DataLoader(self.valid_dataset_all, batch_size=args.gat_batch_size, shuffle=False, num_workers=0)
        self.valid_dataloader_least = DataLoader(self.valid_dataset_least, batch_size=args.gat_batch_size, shuffle=False, num_workers=0)

        self.test_dataset_all = Dataset_HKG(test_data['all']['trajectories'], test_data['all']['length'], test_data['all']['targets'])
        self.test_dataset_least = Dataset_HKG(test_data['least']['trajectories'], test_data['least']['length'], test_data['least']['targets'])
        self.test_dataloader_all = DataLoader(self.test_dataset_all, batch_size=args.gat_batch_size, shuffle=False, num_workers=0)
        self.test_dataloader_least = DataLoader(self.test_dataset_least, batch_size=args.gat_batch_size, shuffle=False, num_workers=0)




    def data_process(self):
        self.train_data_KG = [[relation[0] - 1, relation[1]] for relation in self.train_data_KG if relation[0] != 0]

        inverse_relations = [[relation[0], (relation[1][1], relation[1][0])] for relation in self.train_data_KG if
                             relation[0] == 0]
        self.train_data_KG = self.train_data_KG + inverse_relations

        self.relation_num = max(relation[0] for relation in self.train_data_KG) + 1
        self.entity_num = max([max(relation[1]) for relation in self.train_data_KG])

        self.total_relations = len(self.train_data_KG)
        self.max_arity = max([len(relation[1]) for relation in self.train_data_KG])



    def train(self):
        self.data_process()

        for run in range(self.runs):
            self.model = utils.get_model(self.args, self.train_data_GNN['edges'], self.train_data_KG).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.gat_lr, weight_decay=self.args.gat_weight_decay)
            self.criteria = nn.CrossEntropyLoss()

            if self.args.use_kg:
                self.model.train()
                self.train_kg(run)

            print('-' * 10 + 'Start training GNN' + '-' * 10)
            for epoch in range(self.args.gat_epochs):
                print('-' * 10 +'Training epoch: {}'.format(epoch) + '-' * 10)
                self.model.train()
                self.train_gnn_one_epoch(epoch, run)
                print('-' * 10 +'Epoch {} finished'.format(epoch) + '-' * 10)

                if epoch % self.args.eval_interval == 0:
                    print('-' * 10 + 'Start evaluating at epoch: {}'.format(epoch) + '-' * 10)
                    self.model.eval()
                    self.evaluation(self.valid_dataloader_all, self.valid_dataloader_least, epoch, run)
                    self.evaluation(self.test_dataloader_all, self.test_dataloader_least, epoch, run,
                                    mode='test')
            print('-' * 10 + 'Start testing on test dataset'+ '-' * 10)
            self.save_model(self.args.gat_epochs)



    def train_kg(self, run):
        print('-' * 10 + 'Start training KG' + '-' * 10)
        args = self.args

        optimizer = optim.Adam(self.model.parameters(), lr=args.kg_lr, weight_decay=args.kg_weight_decay)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        global time_start
        time_start = time.time()
        for epoch in range(self.args.kg_epochs):
            random.shuffle(self.train_data_KG)
            iter = 0
            for iteration in range(self.total_relations // self.args.kg_batch_size + 1):
                last_iteration = iteration == self.total_relations // self.args.kg_batch_size
                if last_iteration:
                    print('epoch: {}, iteration: {}, loss: {:.4f}'.format(epoch, iteration, total_loss / iter))
                    self.writer.add_scalar('Runs{}/KG/loss'.format(run), total_loss / 100, epoch * self.total_relations // self.args.kg_batch_size + iteration)
                    total_loss = 0
                    iter = 0
                    continue

                batch_pos = self.train_data_KG[iteration * args.kg_batch_size: (iteration + 1) * args.kg_batch_size]

                batch, labels, ms, bs = sampler.hkg_negative_sample(batch_pos, args.entity_num,
                                                                             neg_ratio=args.neg_ratio,
                                                                             max_arity=self.max_arity)
                number_of_positive = len(np.where(labels > 0)[0])

                batch = torch.LongTensor(batch).to(self.device)
                ms = torch.FloatTensor(ms).to(self.device)
                bs = torch.FloatTensor(bs).to(self.device)

                optimizer.zero_grad()
                predictions = self.model(index=batch, mode='kg', ms=ms, bs=bs)

                predictions = sampler.padd_and_decompose(labels, predictions, args.neg_ratio * self.max_arity)
                targets = torch.zeros(number_of_positive).long().to(self.device)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                iter += 1
                if iteration % 50 == 0:
                    print('epoch: {}, iteration: {}, loss: {:.4f}'.format(epoch, iteration, total_loss / iter))
                    self.writer.add_scalar('Runs{}/KG/loss'.format(run), total_loss / 100, epoch * self.total_relations // self.args.kg_batch_size + iteration)
                    total_loss = 0
                    iter = 0
        print('-' * 10 + 'Training KG Finished' + '-' * 10)

    def train_gnn_one_epoch(self, epoch, run):
        args = self.args

        total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        total_iter = self.train_dataloader.__len__()
        for iteration, inputs in enumerate(self.train_dataloader):
            predictions, labels = self.get_model_prediction(inputs, mode='train')
            loss = self.criteria(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = \
                self.cal_train_metric(total_iter, iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank,
                                      predictions, labels, loss, iteration, epoch, run)



    def get_model_prediction(self, inputs, mode='train') -> tuple:
        trajectories, length, targets = inputs
        if mode == 'train':
            predictions, labels = self.model(index=trajectories, targets=targets, length=length,
                                             mode='train_gat_check_in')
        elif mode == 'test':
            predictions, labels = self.model(index=trajectories, targets=targets, length=length,
                                             mode='test_gat_check_in')
        labels = labels - self.index['poi']['start']
        return predictions, labels


class Trainer_Flashback(Trainer):
    def __init__(self, args, train_data, valid_data, test_data):
        super(Trainer_Flashback, self).__init__(args)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.train_dataloader = DataLoader(self.train_data, batch_size=args.fb_batch_size, shuffle=True)
        self.valid_dataloader_all = DataLoader(self.valid_data['all'], batch_size=args.fb_batch_size, shuffle=False)
        self.valid_dataloader_least = DataLoader(self.valid_data['least'], batch_size=args.fb_batch_size, shuffle=False)
        self.test_dataloader_all = DataLoader(self.test_data['all'], batch_size=args.fb_batch_size, shuffle=False)
        self.test_dataloader_least = DataLoader(self.test_data['least'], batch_size=args.fb_batch_size, shuffle=False)

        self.runs = args.runs

    def train(self):
        args = self.args

        for run in range(self.runs):
            self.model = utils.get_model(args).to(device=self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=args.fb_lr, weight_decay=args.fb_weight_decay)
            self.criteria = nn.CrossEntropyLoss()
            self.h0_strategy = create_h0_strategy(args.fb_hidden_size, args.is_lstm)


            print('-' * 10 + 'Start training {}'.format(self.model_name) + '-' * 10)
            for epoch in range(self.args.fb_epochs):
                print('-' * 10 +'Training epoch: {}'.format(epoch) + '-' * 10)

                self.model.train()
                self.train_one_epoch(epoch, run)
                print('-' * 10 +'Epoch {} finished'.format(epoch) + '-' * 10)

                if epoch % self.args.eval_interval == 0:
                    print('-' * 10 + 'Start evaluating at epoch: {}'.format(epoch) + '-' * 10)
                    self.model.eval()
                    self.evaluation(self.valid_dataloader_all, self.valid_dataloader_least, epoch, run)
                    self.evaluation(self.test_dataloader_all, self.test_dataloader_least, epoch, run, mode='test')

    def train_one_epoch(self, epoch, run):
        args = self.args
        total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        total_iter = self.train_dataloader.__len__()
        for iteration, (x, t, s, y, y_t, y_s, active_users, length) in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()
            predictions, labels = self.get_model_prediction((x, t, s, y, y_t, y_s, active_users, length))
            loss = self.criteria(predictions, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = \
                self.cal_train_metric(total_iter, iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank,
                                    predictions,labels, loss, iteration, epoch, run)

    def get_model_prediction(self, inputs, mode=None) -> tuple:
        x, t, s,y, y_t, y_s, active_users, length = inputs
        h = self.h0_strategy.on_init(x.shape[0], self.device)
        active_users = active_users.squeeze()
        length = length.squeeze()

        x = x.squeeze().permute(1, 0).to(self.device)
        t = t.squeeze().permute(1, 0).to(self.device)
        s = s.squeeze().permute(1, 0, 2).to(self.device)
        y = y.squeeze().to(self.device)
        y_t = y_t.squeeze().to(self.device)
        y_s = y_s.squeeze().to(self.device)
        active_users = active_users.to(self.device)

        x = x - self.index['poi']['start'] # index starts from 0
        x[x < 0] = 0

        predictions = self.model(x, t, s, y_t, y_s, h, active_users, length)
        predictions = predictions.view(-1, self.args.poi_num)
        labels = y.view(-1)
        labels = labels - self.index['poi']['start']

        return predictions, labels
    
class Trainer_Graph_Flashback(Trainer):
    def __init__(self, args, train_data, valid_data, test_data):
        super(Trainer_Graph_Flashback, self).__init__(args)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.kg_train_data = train_data['KG']

        self.train_kg_dataloader = DataLoader(self.kg_train_data, batch_size=args.gfb_batch_size, shuffle=True)
        self.train_dataloader = DataLoader(self.train_data['Flashback'], batch_size=args.gfb_batch_size, shuffle=True)
        self.valid_dataloader_all = DataLoader(self.valid_data['all'], batch_size=args.gfb_batch_size, shuffle=False)
        self.valid_dataloader_least = DataLoader(self.valid_data['least'], batch_size=args.gfb_batch_size, shuffle=False)
        self.test_dataloader_all = DataLoader(self.test_data['all'], batch_size=args.gfb_batch_size, shuffle=False)
        self.test_dataloader_least = DataLoader(self.test_data['least'], batch_size=args.gfb_batch_size, shuffle=False)

        self.runs = args.runs
        self.args = args
        self.args.device = self.device

    def train(self):
        args = self.args

        for run in range(self.runs):
            self.kg_model = utils.get_model(args, mode='kg').to(device=self.device)

            self.criteria = nn.CrossEntropyLoss()
            self.h0_strategy = create_h0_strategy(args.gfb_hidden_size, args.is_lstm)


            print('-' * 10 + 'Start training {}'.format(self.model_name) + '-' * 10)
            print('-' * 10 + 'Training KG phase' + '-' * 10)
            self.train_kg(run)
            print('-' * 10 + 'Training Flashback phase' + '-' * 10)
            # Initialize the model with the trained kg model

            user_encoder = nn.Embedding.from_pretrained(self.kg_model.ent_embeddings.weight[:args.user_num]).to(self.device)
            loc_encoder = nn.Embedding.from_pretrained(self.kg_model.ent_embeddings.weight[args.user_num:]).to(self.device)
            rel_encoder = nn.Embedding.from_pretrained(self.kg_model.rel_embeddings.weight).to(self.device)
            interact_preference = rel_encoder(torch.LongTensor([0]).to(self.device))
            temporal_preference = rel_encoder(torch.LongTensor([1]).to(self.device))
            spatial_preference = rel_encoder(torch.LongTensor([2]).to(self.device))
            friend_preference = rel_encoder(torch.LongTensor([3]).to(self.device))

            args.interact_graph = flashback_graph.construct_user_poi_graph(args, user_encoder, loc_encoder, interact_preference)
            args.spatial_graph = flashback_graph.construct_poi_poi_graph(args, loc_encoder, spatial_preference)
            args.transition_graph = flashback_graph.construct_poi_poi_graph(args, loc_encoder, temporal_preference)
            args.friend_graph = flashback_graph.construct_friend_graph(args, user_encoder, friend_preference)
            self.model = utils.get_model(args).to(device=self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.fb_lr, weight_decay=args.fb_weight_decay)

            for epoch in range(self.args.fb_epochs):
                print('-' * 10 +'Training epoch: {}'.format(epoch) + '-' * 10)

                self.model.train()
                self.train_one_epoch(epoch, run)
                print('-' * 10 +'Epoch {} finished'.format(epoch) + '-' * 10)

                if epoch % self.args.eval_interval == 0:
                    print('-' * 10 + 'Start evaluating at epoch: {}'.format(epoch) + '-' * 10)
                    self.model.eval()
                    self.evaluation(self.valid_dataloader_all, self.valid_dataloader_least, epoch, run)
                    self.evaluation(self.test_dataloader_all, self.test_dataloader_least, epoch, run, mode='test')

    def train_kg(self, run):
        optimizer = optim.Adam(self.kg_model.parameters(), lr=self.args.kg_lr, weight_decay=self.args.kg_weight_decay)
        criteria = lambda pos, neg: torch.sum(torch.max(pos - neg + self.args.gfb_margin, torch.zeros_like(pos)))
        all_triplets = set()
        for h, r, t in self.kg_train_data:
            all_triplets.add((h, r, t))
        
        for epoch in range(self.args.kg_epochs):
            for iteration, (h, r, t) in enumerate(self.train_kg_dataloader):
                h = h.squeeze().to(self.device)
                r = r.squeeze().to(self.device)
                t = t.squeeze().to(self.device)

                h_prime, t_prime = sampler.kg_negtive_sample(self.args, (h, r, t), all_triplets)
                h_prime = h_prime.squeeze().to(self.device)
                t_prime = t_prime.squeeze().to(self.device)
                optimizer.zero_grad()
                pos = self.kg_model(h, r, t)
                neg = self.kg_model(h_prime, r, t_prime)
                loss = criteria(pos, neg)
                loss.backward()
                optimizer.step()
                if iteration % 100 == 0:
                    print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, iteration, loss.item()))

    def train_one_epoch(self, epoch, run):
        args = self.args
        total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        total_iter = self.train_dataloader.__len__()
        for iteration, (x, t, s, y, y_t, y_s, active_users, length) in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()
            predictions, labels = self.get_model_prediction((x, t, s, y, y_t, y_s, active_users, length))
            loss = self.criteria(predictions, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = \
                self.cal_train_metric(total_iter, iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank,
                                    predictions,labels, loss, iteration, epoch, run)

    def get_model_prediction(self, inputs, mode=None) -> tuple:
        x, t, s,y, y_t, y_s, active_users, length = inputs
        h = self.h0_strategy.on_init(x.shape[0], self.device)
        active_users = active_users.squeeze()
        length = length.squeeze()

        x = x.squeeze().permute(1, 0).to(self.device)
        t = t.squeeze().permute(1, 0).to(self.device)
        s = s.squeeze().permute(1, 0, 2).to(self.device)
        y = y.squeeze().to(self.device)
        y_t = y_t.squeeze().to(self.device)
        y_s = y_s.squeeze().to(self.device)
        active_users = active_users.to(self.device)

        x = x - self.index['poi']['start'] # index starts from 0
        x[x < 0] = 0

        predictions = self.model(x, t, s, y_t, y_s, h, active_users, length)

        predictions = predictions.view(-1, self.args.poi_num)
        labels = y.view(-1)
        labels = labels - self.index['poi']['start']

        return predictions, labels

class Trainer_Stan(Trainer):
    def __init__(self, args, train_data, valid_data, test_data):
        super(Trainer_Stan, self).__init__(args)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.train_dataloader = DataLoader(self.train_data, batch_size=args.stan_batch_size, shuffle=True)
        self.valid_dataloader_all = DataLoader(self.valid_data['all'], batch_size=args.stan_batch_size, shuffle=False)
        self.valid_dataloader_least = DataLoader(self.valid_data['least'], batch_size=args.stan_batch_size, shuffle=False)
        self.test_dataloader_all = DataLoader(self.test_data['all'], batch_size=args.stan_batch_size, shuffle=False)
        self.test_dataloader_least = DataLoader(self.test_data['least'], batch_size=args.stan_batch_size, shuffle=False)

        self.runs = args.runs

    def train(self):
        args = self.args

        for run in range(self.runs):
            self.model = utils.get_model(args).to(device=self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=args.stan_lr, weight_decay=args.stan_weight_decay)
            self.criteria = nn.CrossEntropyLoss()


            print('-' * 10 + 'Start training {}'.format(self.model_name) + '-' * 10)
            for epoch in range(self.args.stan_epochs):
                print('-' * 10 +'Training epoch: {}'.format(epoch) + '-' * 10)

                self.model.train()
                self.train_one_epoch(epoch, run)
                print('-' * 10 +'Epoch {} finished'.format(epoch) + '-' * 10)

                if epoch % self.args.eval_interval == 0:
                    print('-' * 10 + 'Start evaluating at epoch: {}'.format(epoch) + '-' * 10)
                    self.model.eval()
                    self.evaluation(self.valid_dataloader_all, self.valid_dataloader_least, epoch, run)
                    self.evaluation(self.test_dataloader_all, self.test_dataloader_least, epoch, run, mode='test')

    def train_one_epoch(self, epoch, run):
        args = self.args
        total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iter = 0
        total_iter = self.train_dataloader.__len__()
        for iteration, inputs in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            predictions, labels = self.get_model_prediction(inputs, mode='train')
            loss = self.criteria(predictions, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank = \
                self.cal_train_metric(total_iter, iter, total_loss, acc_1, acc_5, acc_10, acc_20, mrr, avg_rank,
                                    predictions,labels, loss, iteration, epoch, run)

    def get_model_prediction(self, inputs, mode=None) -> tuple:
        trajectories, lengths, targets, mat1, mat2t = inputs
        trajectories = trajectories.to(self.device)
        lengths = lengths.to(self.device)
        targets = targets.to(self.device)
        mat1 = mat1.to(self.device)
        mat2t = mat2t.to(self.device)
        mat2s = self.train_dataloader.dataset.mat2s

        prob = self.model(trajectories, mat1, mat2s, mat2t, lengths)
        if mode == 'train':
            prob, targets = utils.sampling_prob(prob, targets, self.args.stan_sample_num)
        return prob, targets

