import argparse

from train import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='NYC', help='City name,choices=[NYC, TKY, SP, JK, KL]')
    parser.add_argument('--model', type=str, default='HKGAT', help='Model name, choices=[Flashback, HKGAT, STAN, GraphFlashback]')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--show_iter', type=int, default=40, help='Show loss every show_iter')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU or not')
    parser.add_argument('--cuda', type=int, default=0, help='GPU ID')


    # KG parameters
    parser.add_argument('--kg_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--kg_weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--neg_ratio', type=int, default=10, help='Negative ratio')
    parser.add_argument('--kg_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--kg_batch_size', type=int, default=512, help='KG Batch size')
    parser.add_argument('--use_kg', type=bool, default=True, help='Use KG or not')

    # GAT parameters
    parser.add_argument('--gat_train_iter', type=int, default=200, help='Number of GAT training iterations')
    parser.add_argument('--gat_batch_size', type=int, default=128, help='GAT Batch size')
    parser.add_argument('--gat_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gat_weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--gat_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--rel_emb_dim', type=int, default=128, help='Relation embedding dimension')
    parser.add_argument('--heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--normalization', default='ln', help='Normalization method')
    parser.add_argument('--hid_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--output_heads', type=int, default=1, help='Output heads')
    parser.add_argument('--use_attention', type=bool, default=False, help='Use attention or not')


    # Flashback parameters
    parser.add_argument('--fb_hidden_size', type=int, default=10, help='Hidden dimension of flashback')
    parser.add_argument('--fb_lr', type=float, default=0.01, help='Learning rate of flashback')
    parser.add_argument('--fb_weight_decay', type=float, default=0, help='Weight decay of flashback')
    parser.add_argument('--fb_epochs', type=int, default=20, help='Number of epochs of flashback')
    parser.add_argument('--fb_batch_size', type=int, default=256, help='Batch size of flashback')
    parser.add_argument('--fb_rnn_type', type=str, default='rnn', help='RNN type of flashback, choices=[GRU, LSTM, RNN]')
    parser.add_argument('--is_lstm', type=bool, default=False, help='Use LSTM or not')

    # GrpahFlashback parameters
    parser.add_argument('--gfb_hidden_size', type=int, default=128, help='Hidden dimension of flashback')
    parser.add_argument('--gfb_entity_dim', type=int, default=128, help='Entity dimension of flashback')
    parser.add_argument('--gfb_relation_dim', type=int, default=128, help='Relation dimension of flashback')
    parser.add_argument('--gfb_lr', type=float, default=0.001, help='Learning rate of flashback')
    parser.add_argument('--gfb_kg_lr', type=float, default=0.001, help='Learning rate of flashback kg')
    parser.add_argument('--gfb_kg_epochs', type=int, default=1, help='Number of epochs of flashback kg')
    parser.add_argument('--gfb_weight_decay', type=float, default=0.00001, help='Weight decay of flashback')
    parser.add_argument('--gfb_epochs', type=int, default=20, help='Number of epochs of flashback')
    parser.add_argument('--gfb_batch_size', type=int, default=256, help='Batch size of flashback')
    parser.add_argument('--gfb_L1_flag', type=bool, default=True, help='Whether to use L1 loss in the model')
    parser.add_argument('--gfb_threshold', type=float, default=20, help='Threshold of flashback')
    parser.add_argument('--gfb_margin', type=float, default=1, help='Margin of loss in the model training')

    parser.add_argument('--use_weight', type=bool, default=False, help='Wshether to use weight in the model')
    parser.add_argument('--use_graph_user', type=bool, default=True, help='Whether to use graph user in the model')
    parser.add_argument('--use_spatial_graph', type=bool, default=True, help='Whether to use spatial graph in the model')
    
    
    #STAN parameters
    parser.add_argument('--stan_lr', type=float, default=0.001, help='Learning rate of STAN')
    parser.add_argument('--stan_weight_decay', type=float, default=0, help='Weight decay of STAN')
    parser.add_argument('--stan_epochs', type=int, default=100, help='Number of epochs of STAN')
    parser.add_argument('--stan_batch_size', type=int, default=256, help='Batch size of STAN')
    parser.add_argument('--stan_hidden_size', type=int, default=10, help='Hidden dimension of STAN')
    parser.add_argument('--stan_seq_len', type=int, default=100, help='Sequence length of STAN')
    parser.add_argument('--t_dim', type=int, default=7*24, help='Number of time nodes')
    parser.add_argument('--stan_dropout', type=float, default=0, help='Dropout rate of STAN')
    parser.add_argument('--stan_sample_num', type=int, default=20, help='Number of negative samples in training phase')


    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval')
    parser.add_argument('--ablation', type=bool, default=False, help='Ablation study')

    args = parser.parse_args()
    args.ablation_list = ['HGNN'] # option: social, check_in, side_info， HGNN, ASEQ, onlySEQ

    if args.model == 'HKGAT':
        train_hkgat(args)
    elif args.model == 'STAN':
        train_stan(args)
    elif args.model == 'GraphFlashback':
        train_graphflashback(args)
    elif args.model == 'Flashback':
        train_flashback(args)

if __name__ == '__main__':
    main()
