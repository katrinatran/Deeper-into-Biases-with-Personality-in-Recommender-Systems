#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

Edited 2021

@author: original = Tangrizzly; this code is from the paper "Session-based Recommendation with Graph Neural Networks"

@author: making changes for project = Katrina
        - majority of code is from paper "Session-based Recommendation with Graph Neural Networks"
        - combined with code paper "Personality Bias of Music Recommendation Algorithms"
        - I made them all fit together and added in mann-whitney u test and cross validations

"""

import argparse
import pickle
import time
from utils_gnn import build_graph, Data, split_validation
from model import *
import sys
import os
from tqdm import tqdm
from datetime import datetime
from scipy import stats

sys.path.append(os.path.abspath('../../../'))
from conf import UN_SEEDS, TRAITS, LEVELS
from utils.data_splitter import DataSplitter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='twitter', help='dataset name: twitter/diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=2, help='the number of epochs to train for') # default used to be 30
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  # [0.001, 0.0005, 0.0001] # default used to be 0.001
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def eval_new(model, test_data, high_scores, low_scores, tr, high_idxs: np.ndarray, low_idxs: np.ndarray, tag: str):
    '''
    Performs the evaluation procedure.
    :param preds: predictions
    :param true: true values
    :param high_idxs: indexes for extracting the metrics for the high group
    :param low_idxs: indexes for extracting the metrics for the low group
    :param tag: should be either val or test
    :return: eval_metric, value of the metric considered for validation purposes
            metrics, dictionary of the average metrics for all users, high group, and low group
            metrics_raw, dictionary of the metrics (not averaged) for high group, and low group
    '''

    metrics = dict()
    metrics_raw = dict()

    for lev in LEVELS:

        hit = []
        slices = test_data.generate_batch(model.batch_size)
        for i in slices:
            targets, scores = forward(model, i, test_data)
            sub_scores = scores.topk(lev)[1]

            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                hit.append(np.isin(target-1, score))
        hit = np.array(hit).astype("float32")
        print("LEVEL: " + str(lev) + " HIT")
        print(np.count_nonzero(hit))
        metric_name = "RECALL @ " + str(lev)
        theMax = max(max(high_idxs), max(low_idxs))
        hitSameSize = hit[:theMax]

        high_res = hit[high_idxs]
        low_res = hit[low_idxs]

        if (lev, tr) in high_scores:
            high_scores[(lev, tr)] += [np.mean(high_res)]
            low_scores[(lev, tr)] += [np.mean(low_res)]
        else:
            high_scores[(lev, tr)] = [np.mean(high_res)]
            print(high_scores)
            low_scores[(lev, tr)] = [np.mean(low_res)]
            print(low_scores)

        metrics['{}/{}_at_{}'.format(tag, metric_name, lev)] = np.mean(hitSameSize)
        metrics['{}/high_{}_at_{}'.format(tag, metric_name, lev)] = np.mean(high_res)
        metrics['{}/low_{}_at_{}'.format(tag, metric_name, lev)] = np.mean(low_res)

        metrics_raw['high_{}_at_{}'.format(metric_name, lev)] = high_res
        metrics_raw['low_{}_at_{}'.format(metric_name, lev)] = low_res

    try:
        u_statistic, p_value = mann_whitney_u_test(high_scores[(50, 'ope')], low_scores[(50, 'ope')])
        print("P VALUE FOR MANN WHITNEY U TEST TESTING")
        print(p_value)
    except ValueError:
        print("VALUE ERROR")
        pass
    print("BEFORE METRICS IN EVAL")
    print(metrics)
    return metrics, metrics_raw, high_scores, low_scores

def mann_whitney_u_test(distribution_1, distribution_2):
    """
    Perform the Mann-Whitney U Test, comparing two different distributions.
    Args:
       distribution_1: List.
       distribution_2: List.
    Outputs:
        u_statistic: Float. U statisitic for the test.
        p_value: Float.
    """
    u_statistic, p_value = stats.mannwhitneyu(distribution_1, distribution_2)
    return u_statistic, p_value


def main():
    print('STARTING UNCONTROLLED EXPERIMENTS WITH GNN')
    print('SEEDS ARE: {}'.format(UN_SEEDS))

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    high_scores = {}
    low_scores = {}
    # seed = 7192837
    # if seed == 7192837:
    for seed in tqdm(UN_SEEDS, desc='seeds'):
        UN_LOG_VAL_STR = r'C:\Users\ktran\cs274_project\pers_bias\res\un\{}\{}\val\{}'
        UN_LOG_TE_STR = r'C:\Users\ktran\cs274_project\pers_bias\res\un\{}\{}\test\{}'

        DATA_PATH = '../../../data/inter.csv'
        PERS_PATH = '../../../data/pers.csv'
        UN_OUT_DIR = '../../../data/seed/'

        ds = DataSplitter(DATA_PATH, PERS_PATH, out_dir=UN_OUT_DIR)
        pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path = ds.get_paths(seed)

        # their data is sessionId --> new_user_id, userId, itemId --> new_track_id, timeframe --> play_count, eventdate

        seed_str = str(seed)
        train_file = '../pytorch_code/twitter/' + seed_str + '/train.txt'
        test_file = '../pytorch_code/twitter/' + seed_str + '/test.txt'

        if seed == 6547893:
            n_node = 11120
        elif seed == 2034976:
            n_node = 11225
        elif seed == 2345303:
            n_node = 11206
        elif seed == 7887871:
            n_node = 11038
        elif seed == 1023468:
            n_node = 11350
        elif seed == 8812394:
            n_node = 11298
        elif seed == 2132395:
            n_node = 11279
        elif seed == 4444637:
            n_node = 11078
        elif seed == 7192837:
            n_node = 11005
        elif seed == 6574836:
            n_node = 11179
        elif opt.dataset == 'diginetica':
            n_node = 43098
        elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
            n_node = 37484
        else:
            n_node = 310

        low_high_indxs = dict()
        for tr in TRAITS:
            # This assigns a 4-entry tuple (vd_low_idxs, vd_high_idxs, te_low_idxs, te_high_idxs)
            low_high_indxs[tr] = ds.get_low_high_indxs(pandas_dir_path, uids_dic_path, tr)

        if opt.validation:
            # v_nodes = [11093, 11096, 11088, 11108, 11083]
            # v_nodes = [11235, 11195, 11243, 11223, 11193]
            # v_nodes = [11193, 11169, 11161, 11223, 11093]
            # v_nodes = [11023, 11027, 11010, 10986, 10878]
            # v_nodes = [11337, 11315, 11298, 11320, 11353]
            # v_nodes = [11320, 11306, 11270, 11297, 11304]
            # v_nodes = [11243, 11286, 11219, 11256, 11200]
            # v_nodes = [11068, 10996, 11060, 11049, 10937] # 4444637
            # v_nodes = [11013, 10983, 11044, 10992, 11009] # 7192837
            # v_nodes = [11126, 11125, 11089, 11116, 11060]
            if seed == 7192837:
                v_nodes = [11013, 10983, 11044, 10992, 11009] # 7192837
                num = 1
                for i in range(len(v_nodes)):
                    print("k-fold CROSS VALIDATION @ %ss" % datetime.now())
                    print("RUN OF CROSS VALIDATION: " + str(num))

                    train_file = 'twitter/' + str(seed) + '/kfolds/final/train' + str(i+1) + '.txt'
                    test_file = 'twitter/' + str(seed) + '/kfolds/final/test' + str(i+1) + '.txt'

                    print("LOADING DATA @ %ss" % datetime.now())
                    train_data = pickle.load(open(train_file, 'rb'))
                    test_data = pickle.load(open(test_file, 'rb'))
                    print("DATA LOADED @ %ss" % datetime.now())

                    print("CREATE DATA @ %ss" % datetime.now())
                    train_data = Data(train_data, shuffle=False)
                    test_data = Data(test_data, shuffle=False)
                    print("DATA CREATED @ %ss" % datetime.now())

                    print("TRANS TO CUDA @ %ss" % datetime.now())
                    v_model = trans_to_cuda(SessionGraph(opt, v_nodes[i]))
                    print("TRANS TO CUDA FINISHED @ %ss" % datetime.now())

                    model = train_test(v_model, train_data, test_data)
                    print("BEFORE TRAITS")
                    # Compute metrics at different thresholds
                    full_metrics = dict()
                    full_raw_metrics = dict()
                    for trait in TRAITS:
                        _, _, te_low_idxs, te_high_idxs = low_high_indxs[trait]
                        print(trait)
                        print("BEFORE EVAL")
                        metrics, metrics_raw, high_scores, low_scores = eval_new(model, test_data, high_scores,
                                                                                 low_scores, trait, te_high_idxs,
                                                                                 te_low_idxs, 'test')
                        # Changing the tag for some metrics
                        metrics = {k if ('high' not in k and 'low' not in k) else k[:5] + trait + '_' + k[5:]: v for
                                   k, v in
                                   metrics.items()}
                        metrics_raw = {k if ('high' not in k and 'low' not in k) else trait + '_' + k: v for k, v in
                                       metrics_raw.items()}
                        full_metrics.update(metrics)
                        full_raw_metrics.update(metrics_raw)
                        print('----------------')
                        print(metrics)
                        print("----------------")


        print("LOADING DATA @ %ss" % datetime.now())
        train_data = pickle.load(open(train_file, 'rb'))
        test_data = pickle.load(open(test_file, 'rb'))

        print("DATA LOADED @ %ss" % datetime.now())


        print("CREATE DATA @ %ss" % datetime.now())
        train_data = Data(train_data, shuffle=False)
        test_data = Data(test_data, shuffle=False)

        print("TRANS TO CUDA @ %ss" % datetime.now())
        model = trans_to_cuda(SessionGraph(opt, n_node))
        print("TRANS TO CUDA FINISHED @ %ss" % datetime.now())

        print("STARTING PROCESS @ %ss" % datetime.now())
        start = time.time()
        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)

            # testing
            model = train_test(model, train_data, test_data)
            print("BEFORE TRAITS")
            # Compute metrics at different thresholds
            full_metrics = dict()
            full_raw_metrics = dict()
            for trait in TRAITS:
                _, _, te_low_idxs, te_high_idxs = low_high_indxs[trait]
                print(trait)
                print("BEFORE EVAL")
                metrics, metrics_raw, high_scores, low_scores = eval_new(model, test_data, high_scores, low_scores, trait, te_high_idxs, te_low_idxs, 'test')
                # Changing the tag for some metrics
                metrics = {k if ('high' not in k and 'low' not in k) else k[:5] + trait + '_' + k[5:]: v for k, v in
                            metrics.items()}
                metrics_raw = {k if ('high' not in k and 'low' not in k) else trait + '_' + k: v for k, v in
                                metrics_raw.items()}
                full_metrics.update(metrics)
                full_raw_metrics.update(metrics_raw)
                print('----------------')
                print("AFTER UPDATE")
                print(metrics)
                print("----------------")
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))

    for lev in LEVELS:
        for tr in TRAITS:
            print(str(lev))
            try:
                u_statistic, p_value = mann_whitney_u_test(high_scores[(lev, tr)], low_scores[(lev, tr)])
                print("P VALUE FOR MANN WHITNEY U TEST TESTING")
                print(p_value)
            except ValueError:
                print("VALUE ERROR")
                print(high_scores[(lev, tr)], low_scores[(lev, tr)])
                pass

if __name__ == '__main__':
    main()
