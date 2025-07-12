import argparse
import importlib
import logging
import os
import os.path as osp

import numpy as np
import tqdm
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import config as cfg
from utils import dataset, utils
from utils.base_solver import BaseSolver
from utils.evaluator import CZSL_Evaluator

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--data", type=str, required=True,
                        choices=['MIT', 'UT', 'MITg', 'UTg', 'APY', 'SUN'],
                        help="Dataset name")
    parser.add_argument("--network", type=str, default='symnet',
                        help="Network name (the file name in `network` folder, but without suffix `.py`)")

    parser.add_argument("--epoch", type=int, default=500,
                        help="Maximum epoch during training phase, or epoch to be tested during testing")
    parser.add_argument("--bz", type=int, default=512,
                        help="Train batch size")
    parser.add_argument("--test_bz", type=int, default=1024,
                        help="Test batch size")

    parser.add_argument("--trained_weight", type=str, default=None,
                        help="Restore from a certain trained weight (relative path to './weights')")
    parser.add_argument("--weight_type", type=int, default=1,
                        help="Type of the trained weight: 1-previous checkpoint(default), 2-pretrained classifier, 3-pretrained transformer")

    parser.add_argument("--test_freq", type=int, default=1,
                        help="Frequency of testing (#epoch). Set to 0 to skip test phase")
    parser.add_argument("--snapshot_freq", type=int, default=10,
                        help="Frequency of saving snapshots (#epoch)")

    parser.add_argument("--force", default=False, action='store_true',
                        help="WARINING: clear experiment with duplicated name")

    # model settings

    parser.add_argument("--rmd_metric", type=str, default='softmax',
                        help="Similarity metric in RMD classification")
    parser.add_argument("--distance_metric", type=str, default='L2',
                        help="Distance form")

    parser.add_argument("--obj_pred", type=str, default=None,
                        help="Object prediction from pretrained model")

    parser.add_argument("--wordvec", type=str, default='glove',
                        help="Pre-extracted word vector type")

    # important hyper-parameters

    parser.add_argument("--rep_dim", type=int, default=300,
                        help="Dimentionality of attribute/object representation")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    parser.add_argument("--dropout", type=float,
                        help="Keep probability of dropout")
    parser.add_argument("--batchnorm", default=False, action='store_true',
                        help="Use batch normalization")
    parser.add_argument("--loss_class_weight", default=True, action='store_true',
                        help="Add weight between classes (default=true)")

    parser.add_argument("--fc_att", type=int, default=[512], nargs='*',
                        help="#fc layers after word vector")
    parser.add_argument("--fc_compress", type=int, default=[768], nargs='*',
                        help="#fc layers after hidden layer")
    parser.add_argument("--fc_cls", type=int, default=[512], nargs='*',
                        help="#fc layers in classifiers")

    parser.add_argument("--lambda_cls_attr", type=float, default=0)
    parser.add_argument("--lambda_cls_obj", type=float, default=0)

    parser.add_argument("--lambda_trip", type=float, default=0)
    parser.add_argument("--triplet_margin", type=float, default=0.5,
                        help="Triplet loss margin")

    parser.add_argument("--lambda_sym", type=float, default=0)

    parser.add_argument("--lambda_axiom", type=float, default=0)
    parser.add_argument("--remove_inv", default=False, action="store_true")
    parser.add_argument("--remove_com", default=False, action="store_true")
    parser.add_argument("--remove_clo", default=False, action="store_true")

    parser.add_argument("--no_attention", default=False, action="store_true")

    # not so important
    parser.add_argument("--activation", type=str, default='relu',
                        help="Activation function (relu, elu, leaky_relu, relu6)")
    parser.add_argument("--initializer", type=float, default=None,
                        help="Weight initializer for fc (default=xavier init, set a float number to use Gaussian init)")
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', 'momentum', 'rmsprop'],
                        help="Type of optimizer (sgd, adam, momentum)")

    parser.add_argument("--lr_decay_type", type=str, default='no',
                        help="Type of Learning rate decay: no/exp/cos")
    parser.add_argument("--lr_decay_step", type=int, default=100,
                        help="The first learning rate decay step (only for cos/exp)")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9,
                        help="Decay rate of cos/exp")

    parser.add_argument("--focal_loss", type=float,
                        help="`gamma` in focal loss. default=0 (=CE)")

    parser.add_argument("--clip_grad", default=False, action='store_true',
                        help="Use gradient clipping")
    # [参数部分略，无变化，可复用你的 make_parser() 函数定义]
    return parser

def main():
    logger = logging.getLogger('MAIN')
    parser = make_parser()
    args = parser.parse_args()

    utils.duplication_check(args)
    utils.display_args(args, logger)

    logger.info("Loading dataset")
    train_dataloader = dataset.get_dataloader(args.data, 'train', batchsize=args.bz)
    test_dataloader = dataset.get_dataloader(args.data, 'test', batchsize=args.test_bz, obj_pred=args.obj_pred)

    logger.info("Loading network and solver")
    network = importlib.import_module('network.' + args.network)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = network.Network(train_dataloader, args).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=osp.join(cfg.LOG_ROOT_DIR, args.name))

    sw = SolverWrapper(net, train_dataloader, test_dataloader, args, optimizer, writer, device)
    sw.trainval_model(args.epoch)

class SolverWrapper(BaseSolver):
    def __init__(self, network, train_dataloader, test_dataloader, args, optimizer, writer, device):
        self.network = network
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.writer = writer
        self.device = device

        self.args = args
        self.key_score_name = "score_fc" if args.network == 'fc_obj' else "score_rmd"
        self.criterion = 'real_obj_acc' if args.network == 'fc_obj' else 'top1_acc'

        self.weight_dir = osp.join(cfg.WEIGHT_ROOT_DIR, args.name)
        self.log_dir = osp.join(cfg.LOG_ROOT_DIR, args.name)
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.epoch_num = 0
        self.best_report = defaultdict(dict)

        if args.trained_weight is not None:
            self.trained_weight = os.path.join(cfg.WEIGHT_ROOT_DIR, args.trained_weight)
            if args.weight_type != 1:
                self.clear_folder()
        else:
            self.trained_weight = None
            self.clear_folder()

    def trainval_model(self, max_epoch):
        evaluator = CZSL_Evaluator(self.test_dataloader.dataset, self.network)

        for epoch in range(self.epoch_num + 1, max_epoch + 1):
            self.network.train()
            for batch_ind, batch in tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f'Train Epoch {epoch}'):
                loss = self.network.train_step(batch[9].to(self.device), batch[2].to(self.device), self.optimizer, self.writer, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            if self.args.test_freq > 0 and epoch % self.args.test_freq == 0:
                self.test_and_log(evaluator, epoch)

            if epoch % self.args.snapshot_freq == 0:
                self.snapshot(epoch)

        self.writer.close()

    def test_and_log(self, evaluator, epoch):
        self.network.eval()
        accuracies_pair, accuracies_attr, accuracies_obj = defaultdict(list), defaultdict(list), defaultdict(list)

        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_dataloader, desc='Testing'):
                predictions = self.network.test_step_no_postprocess(batch[4].to(self.device))
                attr_truth = batch[1].to(self.device)
                obj_truth = batch[2].to(self.device)

                for key in predictions.keys():
                    p_pair, p_a, p_o = predictions[key]
                    # pair_results = evaluator.score_model(p_pair, obj_truth)
                    pair_results = {
                        k: (v[0].to(self.device), v[1].to(self.device)) if isinstance(v, tuple)
                        else v.to(self.device) if hasattr(v, 'to') else v
                        for k, v in evaluator.score_model(p_pair, obj_truth).items()
                    }

                    match_stats = evaluator.evaluate_predictions(pair_results, attr_truth, obj_truth)
                    accuracies_pair[key].append(match_stats)
                    a_match, o_match = evaluator.evaluate_only_attr_obj(p_a, attr_truth, p_o, obj_truth)
                    accuracies_attr[key].append(a_match)
                    accuracies_obj[key].append(o_match)

        for name in accuracies_pair.keys():
            accuracies = zip(*accuracies_pair[name])
            accuracies = list(map(torch.cat, accuracies))
            attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, objoracle_acc = map(lambda x: torch.mean(x).item(), accuracies)

            real_attr_acc = torch.mean(torch.cat(accuracies_attr[name])).item()
            real_obj_acc = torch.mean(torch.cat(accuracies_obj[name])).item()

            report_dict = {
                'real_attr_acc': real_attr_acc,
                'real_obj_acc': real_obj_acc,
                'top1_acc': closed_1_acc,
                'top2_acc': closed_2_acc,
                'top3_acc': closed_3_acc,
                'name': self.args.name,
                'epoch': epoch,
            }

            if self.criterion not in self.best_report[name] or report_dict[self.criterion] > self.best_report[name][self.criterion]:
                self.best_report[name] = report_dict

            for key, value in report_dict.items():
                if key not in ['name', 'epoch']:
                    self.writer.add_scalar(f"{name}/{key}", value, epoch)

            print(f"Current {name}: {utils.formated_czsl_result(report_dict)}")
            print(f"Best    {name}: {utils.formated_czsl_result(self.best_report[name])}")

    def snapshot(self, epoch):
        torch.save(self.network.state_dict(), osp.join(self.weight_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    main()
