from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

import sgtapose
import time

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sigmoid12(x):
  y = torch.clamp(x.sigmoid_(), 1e-12)
  return y

def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

def flip_tensor(x):
  return torch.flip(x, [3])
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk_channel(scores, K=100):
  batch, cat, height, width = scores.size()
  
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=100):
  batch, cat, height, width = scores.size()
    
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
  # print('topk_inds', topk_inds)

  topk_inds = topk_inds % (height * width)
  # print('topk_inds_/', topk_inds)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()
#  topk_ys   = (topk_inds / width).float() 
#  topk_xs   = (topk_inds % width).float()
#  print('topk_ys', topk_ys)
    
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  # print('topk_ind', topk_ind)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
      topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  # print('topk_inds', topk_inds)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
  # print((topk_ys * 120 + topk_xs == topk_inds))
  
 
  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

# This code is adapted from
# https://gitlab-master.nvidia.com/pmolchanov/lpr-3d-hand-pose-rgb-demo/blob/master/handpose/models/image_heatmaps_pose2dZrel_softargmax_slim.py
class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=5, learned_beta=False, initial_beta=25.0):
        super(SoftArgmaxPavlo, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        if learned_beta:
            self.beta = torch.nn.Parameter(torch.ones(n_keypoints) * initial_beta)
        else: 
            self.beta = (torch.ones(n_keypoints) * initial_beta).cuda()

    def forward(self, heatmaps, size_mult=1.0):

        epsilon = 1e-8
        bch, ch, n_row, n_col = heatmaps.size()
        n_kpts = ch

        beta = self.beta

        # input has the shape (#bch, n_kpts+1, img_sz[0], img_sz[1])
        # +1 is for the Zrel
        heatmaps2d = heatmaps[:, :n_kpts, :, :]
        heatmaps2d = self.avgpool(heatmaps2d)

        # heatmaps2d has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d.contiguous().view(bch, n_kpts, -1)

        # getting the max value of the maps across each 2D matrix
        map_max = torch.max(heatmaps2d, dim=2, keepdim=True)[0]

        # reducing the max from each map
        # this will make the max value zero and all other values
        # will be negative.
        # max_reduced_maps has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d - map_max

        beta_ = beta.view(1, n_kpts, 1).repeat(1, 1, n_row * n_col)
        # due to applying the beta value, the non-max values will be further
        # pushed towards zero after applying the exp function
        exp_maps = torch.exp(beta_ * heatmaps2d)
        # normalizing the exp_maps by diving it to the sum of elements
        # exp_maps_sum has the shape (#bch, n_kpts, 1)
        exp_maps_sum = torch.sum(exp_maps, dim=2, keepdim=True)
        exp_maps_sum = exp_maps_sum.view(bch, n_kpts, 1, 1)
        normalized_maps = exp_maps.view(bch, n_kpts, n_row, n_col) / (
            exp_maps_sum + epsilon
        )

        col_vals = torch.arange(0, n_col) * size_mult
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col).cuda()
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        row_vals = torch.arange(0, n_row).view(n_row, -1) * size_mult
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col).cuda()
        # row_mat gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_idx = Variable(col_idx, requires_grad=False)
        weighted_x = normalized_maps * col_idx.float()
        weighted_x = weighted_x.view(bch, n_kpts, -1)
        x_vals = torch.sum(weighted_x, dim=2)

        row_idx = Variable(row_idx, requires_grad=False)
        weighted_y = normalized_maps * row_idx.float()
        weighted_y = weighted_y.view(bch, n_kpts, -1)
        y_vals = torch.sum(weighted_y, dim=2)

        out = torch.stack((x_vals, y_vals), dim=2)

        return out

def _softargmaxpavlo(scores):
    batch, cat, height, width = scores.size() 
    loss_func = SoftArgmaxPavlo(cat)
    top_coord = loss_func(scores)[0]
    
    topk_score = torch.zeros(cat)
    topk_clses = torch.arange(cat).view(1, -1)
    
    topk_xs = top_coord[:, 0].view(1, -1).type(torch.int64)
    topk_ys = top_coord[:, 1].view(1, -1).type(torch.int64)
    for i in range(cat):
        topk_score[i] = scores[0][i][topk_ys[0][i]][topk_xs[0][i]]
    topk_score = topk_score.view(batch, -1)
    
    topk_inds = (topk_ys * width + topk_xs).type(torch.int64)
    
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    
def _peaks_info(scores):
#    g1 = time.time()
    batch, cat, height, width = scores.size() 
    # print("scores.size", scores.size())
    peaks_from_belief_maps_kwargs = {}
    peaks_from_belief_maps_kwargs["offset_due_to_upsampling"] = 0.4395
    topk_coord = []
    peaks = sgtapose.image_proc.peaks_from_belief_maps(
        scores[0], **peaks_from_belief_maps_kwargs
        )
    # print("peaks", peaks)
#    g2 = time.time()
#    print("g2 - g1", g2 - g1)
    
    for peak in peaks:
        if len(peak) == 1:
            topk_coord.append([peak[0][0], peak[0][1]])
        else:
            if len(peak) > 1:
                # Try to use the belief map scores
                peak_sorted_by_score = sorted(
                    peak, key=lambda x: x[1], reverse=True
                )
                if (
                    peak_sorted_by_score[0][2] - peak_sorted_by_score[1][2]
                    >= 0.25
                ):
                    # Keep the best score
                    topk_coord.append(
                        [
                            peak_sorted_by_score[0][0],
                            peak_sorted_by_score[0][1],
                        ]
                    )
                else:
                    # Can't determine -- return no detection
                    # Can't use None because we need to return it as a tensor
                    topk_coord.append([-999.999, -999.999])
            else:
                topk_coord.append([-999.999, -999.999])
                                          
#    g3 = time.time()
#    print("g3 - g2", g3 - g2)
    
    topk_score = []
    topk_coord_ad = []
    # print("topk_coord",topk_coord)
    for idx, sample in enumerate(topk_coord):
        this_hm = scores[0][idx]
        if -999.999 in sample:
            topk_score.append(-1.)
            topk_coord_ad.append([0, 0])
        else:
            x, y = sample
            x_int, y_int = int(x), int(y)
            topk_score.append(this_hm[y_int][x_int].cpu())
            topk_coord_ad.append([x_int, y_int])
    # print('topk_score', topk_score)
    # print('topk_coord_ad', topk_coord_ad)
    
#    g4 = time.time()
#    print("g4 - g3", g4 - g3)
    
    topk_clses = torch.arange(cat).view(batch, -1).cuda()
    topk_score_tensor = torch.tensor(topk_score).view(batch, cat)
    topk_coord_ad_tensor = torch.tensor(topk_coord_ad).cuda()
    topk_xs = topk_coord_ad_tensor[:, 0].view(batch, -1).type(torch.int64).cuda()
    topk_ys = topk_coord_ad_tensor[:, 1].view(batch, -1).type(torch.int64).cuda()
    topk_inds = (topk_ys * width + topk_xs).type(torch.int64).cuda()
#    
#    g5 = time.time()
#    print("g5 - g4", g5 - g4)
#    print('topk_ind', topk_inds.device)
#    print('topk_clses', topk_clses.device)
#    print('topk_ys', topk_ys.device)
#    print('topk_xs', topk_xs.device)
    
    return topk_score_tensor, topk_inds, topk_clses, topk_ys, topk_xs


        
    
    
    
    
    
    
    
    
    
    
    
    
    

