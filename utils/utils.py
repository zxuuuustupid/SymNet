import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import logging
from PIL import Image

import time
import argparse

# 假设 config.py 和 aux_data.py 结构不变
from . import config as cfg
from . import aux_data

################################################################################
#                               tools for solvers                              #
################################################################################

def display_args(args, logger, verbose=False):
    """print some essential arguments"""
    if verbose:
        ignore = []
        for k, v in args.__dict__.items():
            if not callable(v) and not k.startswith('__') and k not in ignore:
                logger.info("{:30s}{}".format(k, v))
    else:
        logger.info('Name:       %s' % args.name)
        logger.info('Network:    %s' % args.network)
        logger.info('Data:       %s' % args.data)
        logger.info('FC layers:  At {fc_att}, Cm {fc_compress}, Cls {fc_cls}'.format(**args.__dict__))


def duplication_check(args):
    if args.force:
        return
    elif args.trained_weight is None or args.trained_weight.split('/')[0] != args.name:
        assert not osp.exists(osp.join(cfg.WEIGHT_ROOT_DIR, args.name)), \
            "weight dir with same name exists (%s)" % args.name
        assert not osp.exists(osp.join(cfg.LOG_ROOT_DIR, args.name)), \
            "log dir with same name exists (%s)" % args.name


def formated_czsl_result(report):
    fstr = '[{name}/{epoch}] rA:{real_attr_acc:.4f}|rO:{real_obj_acc:.4f}|Cl/T1:{top1_acc:.4f}|T2:{top2_acc:.4f}|T3:{top3_acc:.4f}'
    return fstr.format(**report)


def formated_multi_result(report):
    fstr = '[{name}/{epoch}] mAUC:{mAUC:.4f} | mAP:{mAP:.4f}'
    return fstr.format(**report)

################################################################################
#                                glove embedder                                #
################################################################################

class Embedder:
    """Word embedding wrapper"""
    def __init__(self, vec_type, vocab, data):
        self.vec_type = vec_type

        if vec_type != 'onehot':
            self.embeds = self.load_word_embeddings(vec_type, vocab, data)
            self.emb_dim = self.embeds.shape[1]
        else:
            self.emb_dim = len(vocab)

    def get_embedding(self, i):
        """Return embedding for index or list of indices"""
        if self.vec_type == 'onehot':
            return F.one_hot(torch.tensor(i), num_classes=self.emb_dim).float()
        else:
            i_onehot = F.one_hot(torch.tensor(i), num_classes=self.embeds.shape[0]).float()
            return torch.matmul(i_onehot, torch.tensor(self.embeds))

    def load_word_embeddings(self, vec_type, vocab, data):
        tmp = aux_data.load_wordvec_dict(data, vec_type)
        if isinstance(tmp, tuple):
            attr_dict, obj_dict = tmp
            attr_dict.update(obj_dict)
            embeds = attr_dict
        else:
            embeds = tmp

        embeds_list = []
        for k in vocab:
            if k in embeds:
                embeds_list.append(embeds[k])
            else:
                raise NotImplementedError('Missing vocab: %s' % k)

        embeds = np.array(embeds_list, dtype=np.float32)
        print('Embeddings shape =', embeds.shape)
        return embeds

################################################################################
#                                network utils                                 #
################################################################################

def repeat_tensor(tensor, axis, multiple):
    """Repeat elements along an axis like numpy.repeat (block-wise)"""
    # Example: repeat_tensor(torch.tensor([[1,2]]), axis=1, multiple=3) => [[1,1,1,2,2,2]]
    tensor = tensor.unsqueeze(axis + 1)  # (B, 1, ...)
    tensor = tensor.expand(*tensor.shape[:axis+1], multiple, *tensor.shape[axis+2:])
    tensor = tensor.reshape(*tensor.shape[:axis], -1, *tensor.shape[axis+2:])
    return tensor

def tile_tensor(tensor, axis, multiple):
    """Tile tensor along axis"""
    return tensor.repeat_interleave(multiple, dim=axis)

def activation_func(name):
    if name == "none":
        return lambda x: x
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "relu":
        return F.relu
    else:
        raise NotImplementedError("activation function %s not implemented" % name)
