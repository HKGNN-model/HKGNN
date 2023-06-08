import numpy as np
import math
import torch




def get_rank(predictions, labels):
    '''
    :param predictions: [batch_size, num_pois]
    :param labels: [batch_size]
    :return: rank
    '''
    sorted_predictions, sorted_indices = torch.sort(predictions, dim=-1, descending=True)
    sorted_indices = sorted_indices.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    ranks = []
    for i in range(0, len(labels)):
        sim_i = sorted_indices[i]
        rank = np.where(sim_i == labels[i])[0][0]
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks



def ACCK(predictions, labels, topK=10):
    '''
    :param predictions: [batch_size, num_pois]
    :param labels: [batch_size]
    :param topK: int
    :return: topK
    '''
    ranks = get_rank(predictions, labels)
    topK = np.mean(ranks < topK)
    return topK


def MRR(predictions, labels):
    '''
    :param predictions: [batch_size, num_pois]
    :param labels: [batch_size]
    :return: MRR
    '''
    ranks = get_rank(predictions, labels)
    mrr = np.mean(1.0 / (ranks + 1.0))
    return mrr

def avgrank(predictions, labels):
    '''
    :param predictions: [batch_size, num_pois]
    :param labels: [batch_size]
    :return: avgrank
    '''
    ranks = get_rank(predictions, labels)
    avgrank = np.mean(ranks)
    return avgrank

