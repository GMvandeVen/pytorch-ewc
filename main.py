#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch
from data import get_dataset, DATASET_CONFIGS
from train import train
from model import MLP
import utils


parser = ArgumentParser('PyTorch Implementation: EWC / Intelligent Synapses')
parser.add_argument('--hidden-size', type=int, default=400)
parser.add_argument('--hidden-layer-num', type=int, default=2)
parser.add_argument('--hidden-dropout-prob', type=float, default=.5)
parser.add_argument('--input-dropout-prob', type=float, default=.2)
parser.add_argument('--task-number', type=int, default=10)
parser.add_argument('--epochs-per-task', type=int, default=1)
parser.add_argument('--lamda', type=float, default=5e+3)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--weight-decay', type=float, default=1e-05)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-size', type=int, default=512)
parser.add_argument('--fisher-estimation-sample-size', type=int, default=1024)
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--consolidate', action='store_true')
parser.add_argument('--plot', default='visdom')
parser.add_argument('--intelligent-synapses', action='store_true')
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--c-weight', type=float, default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # set random seed(s)
    np.random.seed(args.random_seed)

    # generate permutations for the tasks.
    permutations = [
        np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for
        _ in range(args.task_number)
    ]

    # prepare mnist datasets.
    train_datasets = [
        get_dataset('mnist', permutation=p) for p in permutations
    ]
    test_datasets = [
        get_dataset('mnist', train=False, permutation=p) for p in permutations
    ]

    # prepare the model.
    mlp = MLP(
        DATASET_CONFIGS['mnist']['size']**2,
        DATASET_CONFIGS['mnist']['classes'],
        hidden_size=args.hidden_size,
        hidden_layer_num=args.hidden_layer_num,
        hidden_dropout_prob=args.hidden_dropout_prob,
        input_dropout_prob=args.input_dropout_prob,
    )

    # initialize the parameters.
    utils.xavier_initialize(mlp)

    # prepare the cuda if needed.
    if cuda:
        mlp.cuda()

    # run the experiment.
    train(
        mlp, train_datasets, test_datasets,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        test_size=args.test_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lamda=args.lamda,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_log_interval=args.eval_log_interval,
        loss_log_interval=args.loss_log_interval,
        cuda=cuda,
        plot=args.plot,
        pdf_file_name="jnk.pdf",
        epsilon=args.epsilon,
        c=args.c_weight,
        intelligent_synapses=args.intelligent_synapses
    )
