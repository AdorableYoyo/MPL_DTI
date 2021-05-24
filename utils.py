import pandas as pd
import numpy as np
# import pickle
import pickle5 as pickle
import json
from rdkit import Chem


import torch
import argparse

#from ligand_graph_features import *

import os
from datetime import datetime

import json

##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data

##### pickle modules #####
def save_dict_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f)
  return  dict




def load_training_data(exp_path,debug_ratio):
    def load_data(exp_path,file,debug_ratio):
        dataset = pd.read_csv(exp_path +file)
        cut = int(dataset.shape[0] * debug_ratio)
        print(file[:-3] + ' size:', cut)
        return dataset.iloc[:cut,:]

    train = load_data(exp_path,'train.csv',debug_ratio)
    dev   = load_data(exp_path,'dev.csv',debug_ratio)
    test  = load_data(exp_path,'test.csv',debug_ratio)

    return train, dev, test


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


