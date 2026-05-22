# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations
from typing import cast
import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
import torch.nn.functional as F

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)



class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distillation using KL divergence.
    
    This loss measures the divergence between student predictions and teacher
    predictions (soft labels) using KL divergence with temperature scaling.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        loss_weight (float): Weight of the loss. Default: 1.0
        T (float): Temperature for distillation. Higher temperature produces
            softer probability distributions. Default: 10.0
        detach_target (bool): Whether to detach soft labels from the
            computation graph. Default: True
    
    Example:
        >>> loss_fn = KnowledgeDistillationKLDivLoss(T=4.0, loss_weight=0.5)
        >>> student_logits = torch.randn(32, 10)  # (batch_size, num_classes)
        >>> teacher_logits = torch.randn(32, 10)
        >>> loss = loss_fn(student_logits, teacher_logits)
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10.0, detach_target=True):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1, f"Temperature T must be >= 1, got {T}"
        assert reduction in ['none', 'mean', 'sum'], \
            f"reduction must be 'none', 'mean', or 'sum', got {reduction}"
        
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T
        self.detach_target = detach_target
    
    def forward(self, pred, soft_label, weight=None):
        """Forward pass to compute the KL divergence loss.
        
        Args:
            pred (torch.Tensor): Predicted logits from student model.
                Shape: (N, C) or (N, C, H, W) 
            soft_label (torch.Tensor): Target logits from teacher model.
                Must have the same shape as pred.
            weight (torch.Tensor, optional): Element-wise weights.
                Shape should be broadcastable to loss shape.
        
        Returns:
            torch.Tensor: Computed KD loss (scalar if reduction != 'none')
        """
        
        assert pred.size() == soft_label.size(), \
            f"pred and soft_label must have the same shape, got {pred.size()} and {soft_label.size()}"
        
        target = F.softmax(soft_label / self.T, dim=1)
        if self.detach_target:
            target = target.detach()
        
        kd_loss = F.kl_div(
            F.log_softmax(pred / self.T, dim=1),
            target,
            reduction='none'
        )
        kd_loss = kd_loss.sum(dim=1)
        kd_loss = kd_loss.mean(dim=-1) * (self.T ** 2)
        if weight is not None:
            kd_loss = kd_loss * weight
        
        if self.reduction == 'mean':
            kd_loss = kd_loss.mean()
        elif self.reduction == 'sum':
            kd_loss = kd_loss.sum()
        return self.loss_weight * kd_loss

def batch_iou(box1, box2):
    """
    Calculates IoU between two sets of boxes.
    box1, box2: [N, 4] (x1, y1, x2, y2)
    """
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N, 2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N, 2]

    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter / (area1 + area2 - inter + 1e-7)
    return iou



# class FeatureDistillationLoss(nn.Module):
#     def __init__(self, student_channels, teacher_channels, loss_weight=1.0, tau=4.0):
#         super().__init__()
#         self.loss_weight = loss_weight
#         self.tau = tau  
#         self.align_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(s, t, 1, bias=False),
#                 nn.BatchNorm2d(t), 
#             ) if s != t else nn.Identity()
#             for s, t in zip(student_channels, teacher_channels)
#         ])
#         self.level_weights = nn.Parameter(torch.ones(len(student_channels)))
    
#     def forward(self, student_features, teacher_features):
#         total_loss = 0.0
#         for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
#             s_aligned = self.align_layers[i](s_feat)
#             if s_aligned.shape[2:] != t_feat.shape[2:]:
#                 s_aligned = F.interpolate(
#                     s_aligned, size=t_feat.shape[2:], mode='bilinear', align_corners=False
#                 )

#             s_norm = F.normalize(s_aligned, p=2, dim=1)
#             t_norm = F.normalize(t_feat.detach(), p=2, dim=1)
#             #loss = F.mse_loss(s_norm, t_norm, reduction='mean')
#             loss = 1.0 - F.cosine_similarity(s_norm, t_norm, dim=1).mean()

#             weighted_loss = loss * self.level_weights[i]
#             total_loss += weighted_loss

#         avg_loss = total_loss / self.level_weights.sum()
#         return avg_loss * self.loss_weight


# class MaskedGenerativeDistillation(nn.Module):
#     def __init__(
#         self,
#         student_channels: list[int],  # [num_classes+4, num_classes+4, num_classes+4] for 3 scales
#         teacher_channels: list[int],  # 
#         alpha: float = 0.67,
#         loss_weight: float = 2e-6
#     ):
#         super().__init__()
#         assert len(student_channels) == len(teacher_channels)
#         self.alpha = alpha
#         self.loss_weight = loss_weight
#         self.num_levels = len(student_channels)
        
#         self.align_layers = nn.ModuleList([
#             nn.Conv2d(s, t, 1, bias=False) if s != t else nn.Identity()
#             for s, t in zip(student_channels, teacher_channels)
#         ])
        
#         self.generators = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(t, t, 3, padding=1, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(t, t, 3, padding=1, bias=True)
#             ) for t in teacher_channels
#         ])
        
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def generate_mask(self, shape, device):
#         B, C, H, W = shape
#         rand = torch.rand(B, 1, H, W, device=device)
#         mask = (rand >= self.alpha).float()  # Keep (1-alpha) proportion
#         return mask
    
#     def forward(self, student_features, teacher_features):
#         """
#         Args:
#             student_features: List of tensors from student DETECTION HEADS [S^1, S^2, S^3]
#                              Each: (B, C_det, H, W) where C_det = num_classes + 4
#             teacher_features: List of tensors from teacher DETECTION HEADS [T^1, T^2, T^3]
#         """
#         total_loss = 0.0
        
#         for level_idx, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
#             if s_feat.shape[2:] != t_feat.shape[2:]:
#                 s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], 
#                                       mode='bilinear', align_corners=False)
            
#             s_aligned = self.align_layers[level_idx](s_feat)
            
#             mask = self.generate_mask(s_aligned.shape, s_aligned.device)
#             s_masked = s_aligned * mask  
            
#             s_generated = self.generators[level_idx](s_masked)
            
#             level_loss = F.mse_loss(s_generated, t_feat.detachmake_student_hook(), reduction='sum')
#             total_loss += level_loss
        
#         avg_loss = total_loss / self.num_levels
#         return avg_loss * self.loss_weight

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=4.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)
            logsoftmax_s = F.log_softmax(s.view(-1, W * H) / self.tau, dim=1)

            cost = F.kl_div(logsoftmax_s, softmax_pred_T.detach(), reduction='sum') * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss

class MGDLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)
            ).to(device) for channel in teacher_channels
        ])

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss

class ChannelWiseMLP(nn.Module):
    """
    2-layer MLP with ReLU activation for channel-wise transformation.
    Uses 1x1 convolutions as described in the paper.
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels 
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.mlp(x)


class ChannelWiseDistillationLoss(nn.Module):
    """
    Channel-wise Feature Distillation from ICCVW 2023 paper.
    
    "A Simple and Generic Framework for Feature Distillation via Channel-Wise Transformation"
    
    Key differences from CWD/MGD:
    - Uses simple 2-layer MLP (1x1 conv + ReLU + 1x1 conv) only on student features
    - Uses L2 distance instead of KL divergence or generative approach
    - No spatial-wise transformation or attention masks
    - Teacher features are used directly (identity transform)
    """
    def __init__(self, channels_s, channels_t, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.align_modules = nn.ModuleList()
        self.mlp_modules = nn.ModuleList()
        
        for s_ch, t_ch in zip(channels_s, channels_t):
            if s_ch != t_ch:
                align = nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
                nn.init.kaiming_normal_(align.weight, mode='fan_out', nonlinearity='linear')
            else:
                align = nn.Identity()
            
            self.align_modules.append(align)
            self.mlp_modules.append(ChannelWiseMLP(t_ch))
    
    def forward(self, y_s, y_t):
        """
        Args:
            y_s: List of student features [N, C_s, H, W]
            y_t: List of teacher features [N, C_t, H, W]
        Returns:
            L2 distillation loss (scalar)
        """
        assert len(y_s) == len(y_t), f"Number of features mismatch: {len(y_s)} vs {len(y_t)}"
        
        total_loss = 0.0
        n = y_s[0].shape[0]  
        
        
        for idx, (s_feat, t_feat) in enumerate(zip(y_s, y_t)):
            module_device = s_feat.device
            module_dtype  = s_feat.dtype

            self.align_modules[idx] = self.align_modules[idx].to(
                device=module_device,
                dtype=module_dtype
            )

            self.mlp_modules[idx] = self.mlp_modules[idx].to(
                device=module_device,
                dtype=module_dtype
            )

            s_aligned = self.align_modules[idx](s_feat)
            if s_aligned.shape[2:] != t_feat.shape[2:]:
                s_aligned = F.interpolate(
                    s_aligned, size=t_feat.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            s_transformed = self.mlp_modules[idx](s_aligned)
            
            
            layer_loss = F.mse_loss(s_transformed, t_feat.detach(), reduction='mean')            
            total_loss += layer_loss
        

        avg_loss = total_loss / len(y_s)
        return self.loss_weight * avg_loss
    

class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        # Move all modules to same precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Convert to ModuleList and ensure consistent dtype
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        
        # Create alignment modules
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
            
        # Create normalization layers
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        # if distiller == 'mgd':
        #     self.feature_loss = MGDLoss(channels_s, channels_t)
        # elif distiller == 'cwd':
        #     self.feature_loss = CWDLoss(channels_s, channels_t)
        # elif distiller == 'channel_wise' or distiller == 'cw':  # NEW
        #     self.feature_loss = ChannelWiseDistillationLoss(channels_s, channels_t, loss_weight)
        # else:
        #     raise NotImplementedError
        self.feature_loss = ChannelWiseDistillationLoss(channels_s, channels_t, loss_weight)

    def forward(self, y_s, y_t):
        return self.feature_loss(y_s, y_t)
    
        # if len(y_s) != len(y_t):
        #     y_t = y_t[len(y_t) // 2:]

        # tea_feats = []
        # stu_feats = []

        # for idx, (s, t) in enumerate(zip(y_s, y_t)):
            
        #     s = s.type(next(self.align_module[idx].parameters()).dtype)
        #     t = t.type(next(self.align_module[idx].parameters()).dtype)

        #     if self.distiller == "cwd":
        #         s = self.align_module[idx](s)
        #         stu_feats.append(s)
        #         tea_feats.append(t.detach())
        #     else:
        #         t = self.norm[idx](t)
        #         stu_feats.append(s)
        #         tea_feats.append(t.detach())

        # loss = self.feature_loss(stu_feats, tea_feats)
        # return self.loss_weight * loss

class UnifiedDistillationLoss(nn.Module):
    """
    Unified distillation loss combining multiple distillation strategies:
    1. Feature distillation (CWD/MGD) - intermediate layer features
    2. Dense logit distillation (one2many head)
    3. Sparse logit distillation (one2one head)
    4. Box regression distillation
    
    All hyperparameters are configurable via distillation_config dict.
    """
    def __init__(self, models, modelt, distillation_config=None):
        super().__init__()
        self.distillation_config = distillation_config or {}
        self.models = models
        self.modelt = modelt

        self._parse_config()
        self._setup_feature_distillation()
        self._setup_logit_distillation()

        self.teacher_outputs = []
        self.student_outputs = []
        self.remove_handle = []

        self.student_preds = None
        self.teacher_preds = None
    
    def _parse_config(self):
        cfg = self.distillation_config

        self.feature_distiller = cfg.get('feature_distiller', 'cwd')
        self.feature_loss_weight = cfg.get('feature_loss_weight', 1.0)
        self.feature_layers = cfg.get('feature_layers', ["6", "8", "13", "16", "19", "22"])

        self.logit_temperature = cfg.get('logit_temperature', 4.0)
        self.dense_logit_weight = cfg.get('dense_logit_weight', 1.0)
        self.sparse_logit_weight = cfg.get('sparse_logit_weight', 0.5)
        self.logit_detach_target = cfg.get('logit_detach_target', True)
        self.logit_reduction = cfg.get('logit_reduction', 'mean')

        self.box_loss_weight = cfg.get('box_loss_weight', 2.5)
        self.box_objectness_threshold = cfg.get('box_objectness_threshold', 0.3)

        self.dynamic_weight_start = cfg.get('dynamic_weight_start', 1.0)
        self.dynamic_weight_end = cfg.get('dynamic_weight_end', 0.1)

    
    def _setup_feature_distillation(self):
    
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []

        self._find_feature_layers()
        if self.channels_s and self.channels_t:
            self.feature_loss_fn = FeatureLoss(
                channels_s=self.channels_s,
                channels_t=self.channels_t,
                distiller=self.feature_distiller,
                loss_weight=self.feature_loss_weight
            )
            
        else:
            self.feature_loss_fn = None
            LOGGER.warning("No matching layers found for feature distillation")
    
    def _find_feature_layers(self):
        
        def get_layer_output_channels(model, layer_idx):# -> tuple[None, None] | tuple[Any, Any]:
            path = f"model.{layer_idx}"
            module = dict(model.named_modules()).get(path)
            if module is None:
                return None, None
                
            # Infer output channels from module type
            if hasattr(module, 'cv2'):  # C3k2, C2f, etc.
                return module, module.cv2.conv.out_channels
            elif hasattr(module, 'conv'):  # Conv, SPPF
                return module, module.conv.out_channels
            elif hasattr(module, 'out_channels'):  
                return module, module.out_channels
            else:
                next_path = f"model.{layer_idx + 1}"
                next_module = dict(model.named_modules()).get(next_path)
                if hasattr(next_module, 'in_channels'):
                    return module, next_module.in_channels
            return None, None
        
        for idx in self.feature_layers:
            module, channels = get_layer_output_channels(self.modelt, idx)
            if module is not None:
                self.channels_t.append(channels)
                self.teacher_module_pairs.append(module)
                LOGGER.info(f"Teacher layer {idx}: {module.__class__.__name__}, channels={channels}")
        
        for idx in self.feature_layers:
            module, channels = get_layer_output_channels(self.models, idx)
            if module is not None:
                self.channels_s.append(channels)
                self.student_module_pairs.append(module)
                LOGGER.info(f"Student layer {idx}: {module.__class__.__name__}, channels={channels}")

        
        nl = min(len(self.channels_s), len(self.channels_t))
        if nl > 0:
            self.channels_s = self.channels_s[-nl:]
            self.channels_t = self.channels_t[-nl:]
            self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
            self.student_module_pairs = self.student_module_pairs[-nl:]
            
            LOGGER.info(f"Feature distillation: matched {nl} layer pairs using {self.feature_distiller.upper()}")
            for i, (t_ch, s_ch) in enumerate(zip(self.channels_t, self.channels_s)):
                LOGGER.info(f"  Layer {i}: Teacher {t_ch}ch → Student {s_ch}ch")
        
    def _setup_logit_distillation(self):
        """Initialize logit distillation loss functions."""
        self.dense_kd_loss = KnowledgeDistillationKLDivLoss(
            reduction=self.logit_reduction,
            loss_weight=self.dense_logit_weight,
            T=self.logit_temperature,
            detach_target=self.logit_detach_target
        )
        
        self.sparse_kd_loss = KnowledgeDistillationKLDivLoss(
            reduction=self.logit_reduction,
            loss_weight=self.sparse_logit_weight,
            T=self.logit_temperature,
            detach_target=self.logit_detach_target
        )  

    def register_hook(self):
        # Remove the existing hook if they exist
        self.remove_handle_()
        
        self.teacher_outputs = []
        self.student_outputs = []

        def make_student_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    out = output.clone()  # Clone to ensure we don't modify the original
                    l.append(out)
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())  # Detach and clone teacher outputs
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))
        
    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()


    def compute_feature_loss(self):
        """Compute feature distillation loss from captured intermediate features."""
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0)
            
        if len(self.teacher_outputs) != len(self.student_outputs):
            LOGGER.warning(f"Mismatched feature outputs: Teacher={len(self.teacher_outputs)}, Student={len(self.student_outputs)}")
            return torch.tensor(0.0)
            
        loss = self.feature_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs) #type:ignore
        

        return loss 
    
    def compute_logit_loss(self, student_preds, teacher_preds):
        """
        Compute logit distillation losses (dense and sparse).
        
        Args:
            student_preds: Dict with 'one2many' and 'one2one' keys
            teacher_preds: Dict with 'one2many' and 'one2one' keys
        """
        total_loss = torch.tensor(0.0, device=next(self.models.parameters()).device)
        
        s_o2m = student_preds['one2many']
        t_o2m = teacher_preds['one2many']
        s_o2o = student_preds['one2one']
        t_o2o = teacher_preds['one2one']
        s_cls = s_o2m['scores'].permute(0, 2, 1).reshape(-1, s_o2m['scores'].shape[1])
        t_cls = t_o2m['scores'].permute(0, 2, 1).reshape(-1, t_o2m['scores'].shape[1])
        dense_loss = self.dense_kd_loss(s_cls, t_cls)
        total_loss += dense_loss

        s_o2o_cls = s_o2o['scores'].permute(0, 2, 1).reshape(-1, s_o2o['scores'].shape[1])
        t_o2o_cls = t_o2o['scores'].permute(0, 2, 1).reshape(-1, t_o2o['scores'].shape[1])
        sparse_loss = self.sparse_kd_loss(s_o2o_cls, t_o2o_cls)
        total_loss += sparse_loss
        return total_loss
    

    def compute_box_loss(self, student_preds, teacher_preds):
        """
        Compute box regression distillation loss.
        
        Uses teacher confidence to weight the loss (focus on high-confidence predictions).
        """
        s_o2m = student_preds['one2many']
        t_o2m = teacher_preds['one2many']

       
        
        with torch.no_grad():
            t_conf = t_o2m['scores'].max(1)[0].sigmoid()
       
            box_weight = (t_conf > self.box_objectness_threshold).float().unsqueeze(-1)
        
        s_boxes = s_o2m['boxes'].permute(0, 2, 1)  # [B, 8400, 4]
        t_boxes = t_o2m['boxes'].permute(0, 2, 1)  # [B, 8400, 4]
        
        box_loss = (F.smooth_l1_loss(s_boxes, t_boxes, reduction='none') * box_weight).sum() / (box_weight.sum() + 1e-6)

        weighted_loss = box_loss * self.box_loss_weight
        
        return weighted_loss
    
    def set_predictions(self, student_preds, teacher_preds):
        self.student_preds = student_preds
        self.teacher_preds = teacher_preds

    def get_loss(self):
        print()
        if not self.teacher_outputs or not self.student_outputs:
            self.teacher_outputs.clear()
            self.student_outputs.clear()
            self.student_preds = None
            self.teacher_preds = None
            return torch.tensor(0.0, requires_grad=True)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            print(f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}")
            self.teacher_outputs.clear()
            self.student_outputs.clear()
            self.student_preds = None
            self.teacher_preds = None
            return torch.tensor(0.0, requires_grad=True)
        


        device = next(self.models.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        
        # if self.feature_loss_fn is not None:
        #     feat_loss = self.compute_feature_loss()
        #     print(f"{feat_loss=}")
        #     if feat_loss.item() > 0:
        #         total_loss += feat_loss
        
        
        self.teacher_outputs.clear()
        self.student_outputs.clear()

        logit_loss = self.compute_logit_loss(self.student_preds, self.teacher_preds)
        print(f"{logit_loss=}")
        total_loss += logit_loss
    

        box_loss = self.compute_box_loss(self.student_preds, self.teacher_preds)
        total_loss += box_loss
        print(f"{box_loss=}")

        self.student_preds = None
        self.teacher_preds = None


       

        return total_loss
    
    

class BaseTrainer:
    """A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides:
            self.teacher = overrides.get("teacher", None)
            self.distillation_config_loss = overrides.get('distillation_config_loss', None)


            if "teacher" in overrides:
                overrides.pop("teacher")
            
            if 'distillation_config_loss' in overrides:
                overrides.pop('distillation_config_loss')
            
            
            if self.distillation_config_loss and not isinstance(self.distillation_config_loss, dict):
                raise ValueError(f"If distillation_config_loss  is initialized, it must be a dict")
            
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)



        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            # Save run args, serializing augmentations as reprs for resume compatibility
            args_dict = vars(self.args).copy()
            if args_dict.get("augmentations") is not None:
                # Serialize Albumentations transforms as their repr strings for checkpoint compatibility
                args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "args.yaml", args_dict)  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Callbacks - initialize early so on_pretrain_routine_start can capture original args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        


        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()     
            for param in self.teacher.parameters():
                param.requires_grad = False  
            LOGGER.info(f"Knowledge Distillation enabled with teacher model")

        #     self.feat_distill_loss = MGDDistillationLoss(
        #         student_channels=[64, 128, 256],   
        #         teacher_channels=[384, 768, 768],                 
        #     ).to(self.device)
        #     self.model.feat_distill_loss = self.feat_distill_loss
        #     LOGGER.info(f"MGDDistillationLoss enabled: {self.feat_distill_loss}")

          
        

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            print(f"Group {group_idx}:")
            for k, v in param_group.items():
                if k != 'params':
                    print(f"  {k}: {v}")

        
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])


        # distillation_loss_fn = None
        # if hasattr(self, 'teacher') and self.teacher is not None:
        #     kd_config = self.distillation_config_loss or {}
        #     reduction = kd_config.get('reduction', 'mean')
        #     loss_weight = kd_config.get('loss_weight', 1.0)
        #     temperature = kd_config.get('temperature', 10.0)
        #     detach_target = kd_config.get('detach_target', True)
        #     distillation_loss_fn = KnowledgeDistillationKLDivLoss(
        #         reduction=reduction,
        #         loss_weight=loss_weight,
        #         T=temperature,
        #         detach_target=detach_target
        #     )
        #     LOGGER.info(f"KD Loss initialized: T={temperature}, weight={loss_weight}")

        if self.teacher is not None:
            distillation_cfg = getattr(self, 'distillation_config_loss', None) or {}
            self.distillation_loss = UnifiedDistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distillation_config=distillation_cfg
            )
            LOGGER.info(f"Unified distillation enabled with config: {distillation_cfg}")
        else:
            self.distillation_loss = None
    
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            if self.teacher is not None and self.distillation_loss is not None:
                self.distillation_loss.register_hook() 

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    student_preds = self.model(batch["img"])
                    loss, self.loss_items = unwrap_model(self.model).loss(batch, student_preds)
                    self.loss = loss.sum()
                    

                    if self.teacher is not None:
                        progress = (epoch * nb + i) / (self.epochs * nb)
                        distill_weight = 0.5 * (1 + math.cos(progress * math.pi)) 
                        with torch.no_grad():
                            with autocast(self.amp): 
                                teacher_imgs = batch['img']
                                teacher_imgs = teacher_imgs.to(self.device)  
                                teacher_preds = self.teacher(teacher_imgs)
                                            
                        self.distillation_loss.set_predictions(student_preds, teacher_preds[1]) #type:ignore
                            
                        d_loss = cast(UnifiedDistillationLoss, self.distillation_loss).get_loss() 
                        d_loss_weighted = d_loss * distill_weight
                        self.loss = self.loss + d_loss_weighted 
                    
                    if self.teacher is not None:
                        print(f"Task loss: {self.loss.item():.4f}, KD loss: {d_loss.item():.4f}")
                        print(f"Teacher max conf: {teacher_preds[1]['one2many']['scores'].sigmoid().max():.3f}")
                        print(f"Student max conf: {student_preds['one2many']['scores'].sigmoid().max():.3f}")
                                            


                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                
             

                # Backward
                self.scaler.scale(self.loss).backward()

                    
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
            
            if self.teacher is not None:
                self.distillation_loss.remove_handle_()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()

        if self.teacher is not None:
            self.distillation_loss.remove_handle_()
        self.run_callbacks("teardown")
    
    def _compute_distillation_loss(self, student_preds, teacher_preds, distillation_loss_fn, objectness_threshold = 0.3):
        """
        Compute KD loss specifically for YOLO end-to-end / dual-head structures.
        """
        
        # print(f"[DEBUG] - {type(student_preds)=}")
        # print(f"[DEBUG] - {type(teacher_preds)=}")

        # print(f"{student_preds.keys()=}")
        # print(f"{student_preds['one2many'].keys()=}")
        # print(f"{student_preds['one2many']['boxes'].shape=}")
        # print(f"{student_preds['one2many']['scores'].shape=}")
        # print(f"{len(student_preds['one2many']['feats'])=}")
        # print(f"{student_preds['one2many']['feats'][0].shape=}")
        # print(f"{student_preds['one2many']['feats'][1].shape=}")
        # print(f"{student_preds['one2many']['feats'][2].shape=}")

        # print(f"{student_preds['one2one'].keys()=}")
        # print(f"{student_preds['one2one']['boxes'].shape=}")
        # print(f"{student_preds['one2one']['scores'].shape=}")
        # print(f"{len(student_preds['one2one']['feats'])=}")
        # print(f"{student_preds['one2one']['feats'][0].shape=}")
        # print(f"{student_preds['one2one']['feats'][1].shape=}")
        # print(f"{student_preds['one2one']['feats'][2].shape=}")


        # print(f"{len(teacher_preds)=}")
        # print(f"{teacher_preds[1].keys()=}")
        # print(f"{teacher_preds[1]['one2many'].keys()=}")
        # print(f"{teacher_preds[1]['one2many']['boxes'].shape=}")
        # print(f"{teacher_preds[1]['one2many']['scores'].shape=}")
        # print(f"{len(teacher_preds[1]['one2many']['feats'])=}")
        # print(f"{teacher_preds[1]['one2many']['feats'][0].shape=}")
        # print(f"{teacher_preds[1]['one2many']['feats'][1].shape=}")
        # print(f"{teacher_preds[1]['one2many']['feats'][2].shape=}")

      
        # print(f"{teacher_preds[1]['one2one'].keys()=}")
        # print(f"{teacher_preds[1]['one2one']['boxes'].shape=}")
        # print(f"{teacher_preds[1]['one2one']['scores'].shape=}")
        # print(f"{len(teacher_preds[1]['one2one']['feats'])=}")
        # print(f"{teacher_preds[1]['one2one']['feats'][0].shape=}")
        # print(f"{teacher_preds[1]['one2one']['feats'][1].shape=}")
        # print(f"{teacher_preds[1]['one2one']['feats'][2].shape=}")


        total_kd_loss = torch.tensor(0.0, device=self.device)
        t_dict = teacher_preds[1]
        s_dict = student_preds
        
        
        s_o2m, t_o2m = s_dict['one2many'], t_dict['one2many']
        s_o2o, t_o2o = s_dict['one2one'], t_dict['one2one']

      
        # --- 2. DENSE LOGIT KD ---
        s_cls = s_o2m['scores'].permute(0, 2, 1).reshape(-1, s_o2m['scores'].shape[1])
        t_cls = t_o2m['scores'].permute(0, 2, 1).reshape(-1, t_o2m['scores'].shape[1])
        dense_distil_loss =  distillation_loss_fn(s_cls, t_cls)
        total_kd_loss += dense_distil_loss

        # --- 3. SPARSE LOGIT KD (Like-for-Like) ---
        s_o2o_cls = s_o2o['scores'].permute(0, 2, 1).reshape(-1, s_o2o['scores'].shape[1])
        t_o2o_cls = t_o2o['scores'].permute(0, 2, 1).reshape(-1, t_o2o['scores'].shape[1])
        sparse_distil_loss = distillation_loss_fn(s_o2o_cls, t_o2o_cls)
        total_kd_loss += 0.5 * sparse_distil_loss

     
    
        feat_loss = self.feat_distill_loss(s_o2m['feats'], t_o2m['feats'])
        total_kd_loss += feat_loss * 1.0

        # --- 5. BOX REGRESSION (IoU) ---

        with torch.no_grad():
            t_conf = t_o2m['scores'].max(1)[0].sigmoid()
            box_weight = (t_conf > objectness_threshold).float().unsqueeze(-1) 
        
        s_boxes = s_o2m['boxes'].permute(0, 2, 1) # [B, 8400, 4]
        t_boxes = t_o2m['boxes'].permute(0, 2, 1) # [B, 8400, 4]
        
        box_kd = (F.smooth_l1_loss(s_boxes, t_boxes, reduction='none') * box_weight).sum() / (box_weight.sum() + 1e-6)
        total_kd_loss += box_kd * 2.5 
        
        print("\n")
        print(f"{dense_distil_loss=}")
        print(f"{sparse_distil_loss=}")
        print(f"{feat_loss=}")
        print(f"{box_kd=}")
        print("\n")

        return total_kd_loss



    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            # Convert ul:// platform URIs and NDJSON files to local dataset format first
            data_str = str(self.args.data)
            if data_str.endswith(".ndjson") or (data_str.startswith("ul://") and "/datasets/" in data_str):
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo
                from ultralytics.utils.checks import check_file

                self.args.data = str(asyncio.run(convert_ndjson_to_yolo(check_file(self.args.data))))

            # Task-specific dataset checking
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """Run validation on val set using self.validator.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items tensor.

        Notes:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        unwrap_model(self.model).load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
    
        
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                
                fullname = f"kd.{module_name}.{param_name}" if module_name else f"kd.{param_name}"


                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # convert to list of params

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            import re

            # higher lr for certain parameters in MuSGD when funetuning
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
            g_ = []  # new param groups
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer
