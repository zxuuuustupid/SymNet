import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self, dataloader, args, feat_dim=None):
        super(BaseNetwork, self).__init__()

        self.dataloader = dataloader
        self.num_attr = len(dataloader.dataset.attrs)
        self.num_obj = len(dataloader.dataset.objs)

        if feat_dim is not None:
            self.feat_dim = feat_dim
        else:
            self.feat_dim = dataloader.dataset.feat_dim

        self.args = args
        self.dropout = args.dropout if hasattr(args, 'dropout') else None

        if getattr(self.args, 'loss_class_weight', False) and self.args.data not in ['SUN', 'APY']:
            from utils.aux_data import load_loss_weight
            self.num_pair = len(dataloader.dataset.pairs)

            attr_weight, obj_weight, pair_weight = load_loss_weight(self.args.data)
            self.attr_weight = torch.tensor(attr_weight, dtype=torch.float32)
            self.obj_weight = torch.tensor(obj_weight, dtype=torch.float32)
            self.pair_weight = torch.tensor(pair_weight, dtype=torch.float32)
        else:
            self.attr_weight = None
            self.obj_weight = None
            self.pair_weight = None

        # 这里演示定义一个简单MLP作为示例，你实际用法可以继承或替换
        # 输入维度是feat_dim，输出是num_attr + num_obj，具体看需求
        self.mlp = self._make_mlp(self.feat_dim, [512, 256], self.num_attr + self.num_obj)

        # 你可以根据需要定义其他模块、参数

    def _make_mlp(self, input_dim, hidden_layers, output_dim):
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout is not None:
                layers.append(nn.Dropout(p=1 - self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: 输入特征 tensor [batch_size, feat_dim]
        out = self.mlp(x)  # [batch_size, num_attr + num_obj]

        # 假设前半部分是attr预测，后半部分是obj预测
        attr_pred = out[:, :self.num_attr]
        obj_pred = out[:, self.num_attr:]

        # 你可以根据实际情况返回不同东西
        return attr_pred, obj_pred

    def cross_entropy(self, prob, labels, weight=None, sample_weight=None, neg_loss=0, focal_loss_gamma=None):
        device = prob.device
        labels_onehot = F.one_hot(labels, num_classes=prob.shape[1]).float()
        return self.cross_entropy_with_labelvec(prob, labels_onehot, weight, sample_weight, neg_loss, focal_loss_gamma)

    def cross_entropy_with_labelvec(self, prob, label_vecs, weight=None, sample_weight=None, neg_loss=0, focal_loss_gamma=None, mask=None):
        epsilon = 1e-8
        alpha = neg_loss
        gamma = focal_loss_gamma

        pos_mask = label_vecs > 0
        neg_mask = label_vecs == 0

        pos_prob = torch.where(pos_mask, prob, torch.ones_like(prob))
        neg_prob = torch.where(pos_mask, torch.zeros_like(prob), prob)

        pos_xent = -torch.log(pos_prob.clamp(min=epsilon))
        if gamma is not None:
            pos_xent = ((1 - pos_prob) ** gamma) * pos_xent

        if alpha is None or alpha == 0:
            neg_xent = torch.zeros_like(pos_xent)
        else:
            neg_xent = -alpha * torch.log((1 - neg_prob).clamp(min=epsilon))
            if gamma is not None:
                neg_xent = (neg_prob ** gamma) * neg_xent

        xent = pos_xent + neg_xent

        if mask is not None:
            xent = xent * mask

        if sample_weight is not None:
            xent = xent * sample_weight.unsqueeze(1)

        xent = xent.mean(dim=0)

        if weight is not None:
            xent = xent * weight

        return xent.sum()

    def distance_metric(self, a, b):
        if self.args.distance_metric == 'L2':
            return torch.norm(a - b, dim=-1)
        elif self.args.distance_metric == 'L1':
            return torch.norm(a - b, p=1, dim=-1)
        elif self.args.distance_metric == 'cos':
            a_norm = F.normalize(a, p=2, dim=-1)
            b_norm = F.normalize(b, p=2, dim=-1)
            return torch.sum(a_norm * b_norm, dim=-1)
        else:
            raise NotImplementedError(f"Unsupported distance metric: {self.args.distance_metric}")

    def mse_loss(self, a, b):
        return torch.mean(self.distance_metric(a, b))

    def calibration_look_up(self, prob, transform_attr_onehot, gt=1, margin=0.05):
        if not hasattr(self, 'att_sim'):
            raise AttributeError("self.att_sim not defined")

        if gt == 1:
            prob_label_vec = self.att_sim
        elif gt == 0:
            prob_label_vec = 1 - self.att_sim
        else:
            raise ValueError(f"gt should be 0 or 1 but got {gt}")

        if (prob_label_vec < 0).any():
            prob_label_vec = prob_label_vec * 0.5 + 0.5

        carlib_prob = torch.matmul(transform_attr_onehot.float(), prob_label_vec.float())
        prob_diff = torch.abs(prob - carlib_prob)
        loss = torch.clamp(prob_diff - margin, min=0)
        return loss.mean()

    def triplet_margin_loss(self, anchor, positive, negative, weight=None, margin=None):
        d1 = self.distance_metric(anchor, positive)
        d2 = self.distance_metric(anchor, negative)
        return self.triplet_margin_loss_with_distance(d1, d2, weight, margin)

    def triplet_margin_loss_with_distance(self, d1, d2, weight=None, margin=None):
        if margin is None:
            margin = getattr(self.args, 'triplet_margin', 1.0)
        if weight is None:
            dist = d1 - d2 + margin
        else:
            dist = weight * (d1 - d2) + margin
        return torch.clamp(dist, min=0).mean()

    def test_step(self, inputs, device):
        dset = self.dataloader.dataset

        test_att = torch.tensor([dset.attr2idx[attr] for attr, _ in dset.pairs], device=device)
        test_obj = torch.tensor([dset.obj2idx[obj] for _, obj in dset.pairs], device=device)

        pos_image_feat = inputs[4].to(device)

        self.eval()
        with torch.no_grad():
            attr_pred, obj_pred = self.forward(pos_image_feat)
            # 你可以在这里把attr_pred,obj_pred组合成score字典或其他格式
            score = {
                'attr_pred': attr_pred,
                'obj_pred': obj_pred,
            }
        return score
