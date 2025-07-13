import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from network.base_network import BaseNetwork
import logging


class Network(BaseNetwork):
    root_logger = logging.getLogger("network")

    def __init__(self, dataloader, args, feat_dim=None):
        super(Network, self).__init__(dataloader, args, feat_dim)

        # 定义 MLP 分类器（用于训练和测试）
        self.obj_classifier = self._make_mlp(self.feat_dim, args.fc_cls, self.num_obj)
        # 用于tensorboard或记录loss
        self.loss_log = []

    def build_network(self, input_feats, obj_labels=None, test_only=False):
        """
        input_feats: torch.Tensor, shape = (B, feat_dim)
        obj_labels: torch.Tensor, shape = (B,) 可选
        test_only: bool
        """


        # logger = self.logger('create_train_arch')
        batch_size = input_feats.size(0)

        # 训练阶段 forward
        score_pos_O = self.obj_classifier(input_feats)  # [B, num_obj]
        prob_pos_O = F.softmax(score_pos_O, dim=1)

        loss = None
        if not test_only and obj_labels is not None:
            loss = self.cross_entropy(prob_pos_O, obj_labels, weight=self.obj_weight.to(input_feats.device))

        # 推理阶段结构
        prob_pos_A = torch.zeros(batch_size, self.num_attr, device=input_feats.device)
        score_original = torch.zeros(batch_size, self.num_pair, device=input_feats.device)

        score_res = OrderedDict([
            ("score_fc", [score_original, prob_pos_A, prob_pos_O]),
        ])

        if not test_only:
            return (loss, score_res)
        else:
            batch_size = input_feats.size(0)
            prob_pos_A = torch.zeros(batch_size, self.num_attr, device=input_feats.device)
            score_original = torch.zeros(batch_size, self.num_pair, device=input_feats.device)

            score_res = OrderedDict([
                ("score_fc", [score_original, prob_pos_A, prob_pos_O]),
            ])
            return score_res

    def train_step(self, input_feats, obj_labels, optimizer=None, writer=None, global_step=0):
        """
        input_feats: (B, feat_dim)
        obj_labels: (B,)
        optimizer: torch.optim
        """
        self.train()
        loss, _ = self.build_network(input_feats, obj_labels, test_only=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None:
            writer.add_scalar("loss_total", loss.item(), global_step)

        return loss.item()

    def test_step_no_postprocess(self, input_feats):
        """
        input_feats: (B, feat_dim)
        return: dict of scores
        """

        self.eval()
        with torch.no_grad():
            score_res = self.build_network(input_feats, test_only=True)
        return score_res
