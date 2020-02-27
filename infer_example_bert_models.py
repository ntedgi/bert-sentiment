# -*- coding: utf-8 -*-
# file: infer_example_bert_models.py
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from models.lcf_bert import LCF_BERT
from models.aen import AEN_BERT
from models.bert_spc import BERT_SPC
from pytorch_transformers import BertModel
from data_utils import Tokenizer4Bert
import argparse
import csv
import sys
import time


def write_log_to_file():
    logfile = '/home/naort/Desktop/Sentiment/ABSA-PyTorch/log.txt'
    sys.stdout = open(logfile, 'w')


def loadCSV(file):
    result = []
    with open(file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            result.append(row)
    return result


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def prepare_data(text_left, aspect, text_right, tokenizer):
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()

    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence(
        '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    return opt


def predict(left, aspect, right, filters=[]):
    def applyFilter(txt, filters):
        current_str = txt
        for filterFN in filters:
            current_str = filterFN(current_str)
        return current_str

    left = applyFilter(left, filters)
    right = applyFilter(right, filters)

    text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data(left, aspect, right, tokenizer)
    text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
    bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
    text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
    aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
    inputs = [text_bert_indices, bert_segments_ids]
    outputs = model(inputs)
    t_probs = F.softmax(outputs, dim=-1).cpu().numpy()

    # print(f'left | {left}')
    # print(f'aspect | {aspect}')
    # print(f'right | {right}')
    # print('t_probs = ', t_probs)
    sentiment = t_probs.argmax(axis=-1) - 1
    # print('aspect sentiment = ', sentiment)
    # print('=========================')
    return sentiment.max(), t_probs


def removeURL(text):
    parts = text.split(" ")
    parts = list(filter(lambda x: "http" not in x, parts))
    return " ".join(parts)


def removeMentions(text):
    parts = text.split(" ")
    parts = list(filter(lambda x: "@" not in x, parts))
    return " ".join(parts)


def removeHashTags(text):
    parts = text.split(" ")
    parts = list(filter(lambda x: "#" not in x, parts))
    return " ".join(parts)


def splitContent(text, aspect):
    parts = text.split(aspect)
    if len(parts) == 2:
        result = parts[0], aspect, parts[1]
    else:
        raise Exception("error in prarsing")
    return result


def processFile(inputFile, aspect):
    input = loadCSV(inputFile)
    result = []
    first = False
    totalErrors = 0
    for post in input:
        if first:
            try:
                parts = splitContent(post[1], aspect)
                result.append([post[0], parts])
            except:
                # print(f' err in post {post[0]}')
                totalErrors += 1
        else:
            first = True
    print(f'total errors : {totalErrors}')
    return result





if __name__ == '__main__':
    global opt
    global opt
    model_classes = {
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT
    }
    # set your trained models here
    state_dict_paths = {
        'lcf_bert': 'state_dict/lcf_bert_laptop_val_acc0.2492',
        'bert_spc': 'state_dict/bert_spc_restaurant_val_acc0.8509',
        'aen_bert': 'state_dict/aen_bert_laptop_val_acc0.2006'
    }
    opt = get_parameters()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = model_classes[opt.model_name](bert, opt).to(opt.device)
    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(state_dict_paths[opt.model_name]))
    model.eval()
    torch.autograd.set_grad_enabled(False)

    inputFile = "/home/naort/Desktop/Sentiment/ABSA-PyTorch/datasets/clinton negative exmple.csv"
    input = loadCSV(inputFile)
    input.pop(0)
    filtersSet = [[removeURL, removeMentions, removeHashTags]]

    millis = int(round(time.time() * 1000))
    for filters in filtersSet:
        negative = []
        positive = []
        natural = []
        print(f'FILTER IN CURRENT RUN : {filters}')
        print(f'#########################################################################')
        for row in input:
            sentiment, t_probs = predict(row[0], row[1], row[2], filters)
            if sentiment == 1:
                positive.append(t_probs.max())
            if sentiment == -1:
                negative.append(t_probs.max())
            if sentiment == 0:
                natural.append(t_probs.max())
        pos = np.array(positive)
        neg = np.array(negative)
        nat = np.array(natural)
        print('finish predicting files.')
        print('=========================')
        print(f'total posts : {len(natural) + len(positive) + len(negative)} ')
        print(f'total positive : avg({pos.sum() / len(pos)}) std({pos.std()})   {len(positive)} | {positive}')
        print(f'total negative : avg({neg.sum() / len(neg)}) std({neg.std()}) {len(negative)} | {negative}')
        print(f'total natural : avg({nat.sum() / len(nat)}) std({nat.std()}) {len(natural)} | {natural}')
        print('=========================')
    now = int(round(time.time() * 1000))
    print(f'total time for 25 {now - millis} avg({25 / (now - millis)})')
    millis = now
