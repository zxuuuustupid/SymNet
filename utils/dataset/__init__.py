from .. import config as cfg
from utils.dataset import CZSL_dataset, GCZSL_dataset, Multi_dataset
from torch.utils.data import DataLoader
import numpy as np
# import Multi_dataset

import torch

def custom_collate_fn(data):
    result = []
    for d in zip(*data):
        if any(x is None for x in d):
            result.append(None)
        else:
            result.append(torch.stack([torch.tensor(x) for x in d], dim=0))
    return result

def get_dataloader(dataset_name, phase, feature_file="features.t7", batchsize=1, num_workers=1, shuffle=None,args=None, **kwargs):
    
    if dataset_name in ['APY','SUN']:
        dataset = Multi_dataset.MultiDatasetActivations(
            data = dataset_name,
            phase=phase,
            args= args
        )
    elif dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
        dataset =  GCZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = cfg.GCZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            **kwargs)
    else:
        dataset =  CZSL_dataset.CompositionDatasetActivations(
            name = dataset_name,
            root = cfg.CZSL_DS_ROOT[dataset_name], 
            phase = phase,
            feat_file = feature_file,
            **kwargs)
    

    if shuffle is None:
        shuffle = (phase=='train')
    
    return DataLoader(dataset, batchsize, shuffle, num_workers=num_workers,
        collate_fn = custom_collate_fn
    )


    

