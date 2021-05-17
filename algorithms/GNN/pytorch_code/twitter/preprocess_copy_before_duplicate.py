#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

Edited 2021

@author: original = Tangrizzly; making changes for project = Katrina
"""

import argparse
import pickle
from model import *
import sys
import os
from tqdm import tqdm
from datetime import datetime
import csv
import operator


sys.path.append(os.path.abspath('../../../../../'))
from conf import UN_SEEDS, TRAITS
from utils.data_splitter import DataSplitter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
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


def main():
    print('STARTING UNCONTROLLED PREDATA PROCESSING EXPERIMENTS WITH GNN')
    print('SEEDS ARE: {}'.format(UN_SEEDS))

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for seed in tqdm(UN_SEEDS, desc='seeds'):
        UN_LOG_VAL_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\un\{}\{}\val\{}'
        UN_LOG_TE_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\un\{}\{}\test\{}'

        DATA_PATH = '../../../../../data/inter.csv'
        PERS_PATH = '../../../../../data/pers.csv'
        UN_OUT_DIR = '../../../../../data/seed/'

        log_val_str = UN_LOG_VAL_STR.format('gnn', now, seed)
        log_te_str = UN_LOG_TE_STR.format('gnn', now, seed)

        ds = DataSplitter(DATA_PATH, PERS_PATH, out_dir=UN_OUT_DIR)
        pandas_dir_path, scipy_dir_path, uids_dic_path, tids_path = ds.get_paths(seed)

        # this dataset is the training data
        dataset_tr = os.path.join(pandas_dir_path, 'te_tr_data.csv')
        dataset_te = os.path.join(pandas_dir_path, 'te_te_data.csv')

        # their data is sessionId --> new_user_id, userId, itemId --> new_track_id, timeframe --> play_count, eventdate

        def make_tr_dataset(dataset):
            print("-- Starting @ %ss" % datetime.now())
            with open(dataset, "r") as f:
                reader = csv.DictReader(f, delimiter=',')
                sess_clicks = {}
                ctr = 0
                tra_sess = []
                for data in reader:
                    userid = data['new_user_id']
                    tra_sess += [userid]
                    item = data['new_track_id'], (data['play_count'])
                    if userid in sess_clicks:
                        sess_clicks[userid] += [item]
                    else:
                        sess_clicks[userid] = [item]
                    ctr += 1
                for i in list(sess_clicks):
                    sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                    sess_clicks[i] = [c[0] for c in sorted_clicks]
            print("-- Reading data @ %ss" % datetime.now())

            # WHERE SPLIT STARTS

            # Choosing item count >=5 gives approximately the same number of items as reported in paper
            item_dict = {}

            # Convert training sessions to sequences and renumber items to start from 1
            def obtian_tra():
                train_ids = []
                train_seqs = []
                item_ctr = 1
                for s in tra_sess:
                    seq = sess_clicks[s]
                    outseq = []
                    for i in seq:
                        if i in item_dict:
                            outseq += [item_dict[i]]
                        else:
                            outseq += [item_ctr]
                            item_dict[i] = item_ctr
                            item_ctr += 1
                    if len(outseq) < 2:  # Doesn't occur
                        continue
                    train_ids += [s]
                    train_seqs += [outseq]
                print(item_ctr)  # 43098, 37484
                return train_ids, train_seqs

            tra_ids, tra_seqs = obtian_tra()

            def process_seqs(iseqs):
                out_seqs = []
                labs = []
                ids = []
                for id, seq in zip(range(len(iseqs)), iseqs):
                    for i in range(1, len(seq)):
                        tar = seq[-i]
                        labs += [tar]
                        out_seqs += [seq[:-i]]
                        ids += [id]
                return out_seqs, labs, ids

            tr_seqs, tr_labs, tr_ids = process_seqs(tra_seqs)
            tra = (tr_seqs, tr_labs)

            return tra, tra_seqs, item_dict

        def make_te_dataset(dataset, item_dict):
            print("-- Starting @ %ss" % datetime.now())
            with open(dataset, "r") as f:
                reader = csv.DictReader(f, delimiter=',')
                sess_clicks = {}
                ctr = 0
                tes_sess = []
                for data in reader:
                    userid = data['new_user_id']
                    tes_sess += [userid]
                    item = data['new_track_id'], (data['play_count'])
                    if userid in sess_clicks:
                        sess_clicks[userid] += [item]
                    else:
                        sess_clicks[userid] = [item]
                    ctr += 1
                for i in list(sess_clicks):
                    sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                    sess_clicks[i] = [c[0] for c in sorted_clicks]
            print("-- Reading data @ %ss" % datetime.now())

            # Convert test sessions to sequences, ignoring items that do not appear in training set
            def obtian_tes():
                test_ids = []
                test_seqs = []
                for s in tes_sess:
                    seq = sess_clicks[s]
                    outseq = []
                    for i in seq:
                        if i in item_dict:
                            outseq += [item_dict[i]]
                    # if len(outseq) < 2:
                    #     continue
                    test_ids += [s]
                    test_seqs += [outseq]
                return test_ids, test_seqs

            tes_ids, tes_seqs = obtian_tes()

            def process_seqs(iseqs):
                out_seqs = []
                labs = []
                ids = []
                for id, seq in zip(range(len(iseqs)), iseqs):
                    for i in range(1, len(seq)):
                        tar = seq[-i]
                        labs += [tar]
                        out_seqs += [seq[:-i]]
                        ids += [id]
                return out_seqs, labs, ids

            te_seqs, te_labs, te_ids = process_seqs(tes_seqs)
            tes = (te_seqs, te_labs)
            return tes

        print("-- Splitting train set and test set @ %ss" % datetime.now())

        tra, tra_seqs, item_dict = make_tr_dataset(dataset_tr)
        tes = make_te_dataset(dataset_te, item_dict)

        new_dir_name = 'twitter/' + str(seed)
        if not os.path.exists('../../datasets/twitter'):
            os.makedirs('../../datasets/twitter')
        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)
        pickle.dump(tra, open(new_dir_name + '/train.txt', 'wb'))
        pickle.dump(tes, open(new_dir_name + '/test.txt', 'wb'))
        pickle.dump(tra_seqs, open(new_dir_name + '/all_train_seq.txt', 'wb'))
        print("DONE!")


        # current values: sess_clicks (id again {id: # clicks}), sess_date (id {id: date}), ctr (counter #),
        # curid (current ses id #), curdate (time), iid_counts (number times item appears {})
        # tra_sess (training set = session ids), tes_sess (test set)
        # going to dump into tra (train.txt), tes (test.txt), tra_seqs (all_train_seq.txt)

        # our data is user_id, new_track_id, play_count, new_user_id
        # their data is sessionId --> new_user_id, userId, itemId --> new_track_id, timeframe, eventdate --> play_count

        # tra_seqs --> from obtian_tra (tra_sess, sess_clicks, item_dict{empty start})
        #       tra_seqs = obtian_tra's train_seqs
        #       train_seqs = [] of outseq
        #       outseq --> item_ctr --> which is like the item # we started
        #       item_ctr --> starts at 1 in beginning of obtian_tra
        #       item_dict[i] = item_ctr --> item_dict[itemID] = itemctr --> new itemId?
        #       i = i in seq (for each itemId in session_id)
        #       seq = sess_clicks[s] --> sess_clicks[sessionId] = [itemId (song_id), date (num_plays?)]
        #       s = in tra_sess --> session_id

        # tra --> (tr_seqs, tr_labs)
        #   (tr_seqs, tr_labs) = from process_seqs(tra_seqs, tra_dates)
        #   process_seqs = out_seqs, labs
        #       out_seqs = [seq backwards] --> seq[:-i] means seq but cut off i from the end
        #       out_seqs = [[full seq]. [full seq - 1]]...
        #       labs = [seq[-i]] --> seq[ith item from end] --> opposite from out_seq

        # tes --> (te_seqs, te_labs)
        #   same from process_seqs(tes_seqs, tes_dates)
        #   tes_seqs = obtian_tes()
        #       obtain_tes --> want test_seqs
        #       test_seqs


if __name__ == '__main__':
    main()
