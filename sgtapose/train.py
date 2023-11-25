# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:20:27 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset
import torch.utils.data 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

from lib.opts_parallel import opts
from lib.model.model import create_model, load_model, save_model, create_dream_hourglass
from utilities import find_ndds_seq_data_in_dir, set_random_seed, exists_or_mkdir
from datasets import CenterTrackSeqDataset

from lib.trainer_parallel import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from Dream_ct_inference import inference, inference_real
import json


def get_optimizer(opt, model):
    if opt.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == 'sgd':
      print('Using SGD')
      optimizer = torch.optim.SGD(
        model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
    else:
      assert 0, opt.optim
    return optimizer

def save_results(train_log, kp_metrics, pnp_results, mode, writer, epoch):
    train_log[mode] = {}
    train_log[mode]["kp_metrics"] = {}
    train_log[mode]["kp_metrics"]["correctness"] = []
    train_log[mode]["pnp_results"] = {}
    train_log[mode]["pnp_results"]["correctness"] = []
    
    # save kp_metrics results
    if kp_metrics["num_gt_outframe"] > 0:
        out_of_frame_not_found_rate = "Percentage out-of-frame gt keypoints not found (correct): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_missing_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0,
                        kp_metrics["num_missing_gt_outframe"],
                        kp_metrics["num_gt_outframe"],
                    )
        out_of_frame_found_rate = "Percentage out-of-frame gt keypoints found (incorrect): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_found_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0,
                        kp_metrics["num_found_gt_outframe"],
                        kp_metrics["num_gt_outframe"],
                    )
        writer.add_scalar(f"{mode}/out_of_frame_not_found_rate (correct)", round(float(kp_metrics["num_missing_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0, 3), epoch)
        writer.add_scalar(f"{mode}/out_of_frame_found_rate (incorrect)", round(float(kp_metrics["num_found_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0, 3), epoch)
        
    else:
        out_of_frame_not_found_rate = None
        out_of_frame_found_rate = None
    
    train_log[mode]["kp_metrics"]["correctness"] += [out_of_frame_not_found_rate, out_of_frame_found_rate]
    if kp_metrics["num_gt_inframe"] > 0:
        in_frame_not_found_rate = "Percentage in-frame gt keypoints not found (incorrect): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_missing_gt_inframe"])
                        / float(kp_metrics["num_gt_inframe"])
                        * 100.0,
                        kp_metrics["num_missing_gt_inframe"],
                        kp_metrics["num_gt_inframe"],
                    )
        in_frame_found_rate = "Percentage in-frame gt keypoints found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                )
                
        writer.add_scalar(f"{mode}/in_frame_not_found_rate (incorrect)", round(float(kp_metrics["num_missing_gt_inframe"])
                        / float(kp_metrics["num_gt_inframe"])
                        * 100.0, 3), epoch)
        writer.add_scalar(f"{mode}/in_frame_found_rate (correct)", round(float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0, 3), epoch)
        
        train_log[mode]["kp_metrics"]["correctness"] += [in_frame_not_found_rate, in_frame_found_rate]
        if kp_metrics["num_found_gt_inframe"] > 0:
            L2_info = "L2 error (px) for in-frame keypoints (n = {}):".format(
                        kp_metrics["num_found_gt_inframe"]
                    )
            kp_AUC = "   AUC: {:.5f}".format(kp_metrics["l2_error_auc"])
            kp_AUC_threshold = " AUC threshold: {:.5f}".format(
                        kp_metrics["l2_error_auc_thresh_px"]
                    )
            Mean_pck = "   Mean: {:.5f}".format(kp_metrics["l2_error_mean_px"])
            Median_pck = "   Median: {:.5f}".format(kp_metrics["l2_error_median_px"])
            std_pck = "   Std Dev: {:.5f}".format(kp_metrics["l2_error_std_px"])
            train_log[mode]["kp_metrics"]["results"] = {}
            train_log[mode]["kp_metrics"]["results"]["L2_info"] = L2_info
            train_log[mode]["kp_metrics"]["results"]["kp_AUC"] = kp_AUC
            train_log[mode]["kp_metrics"]["results"]["kp_AUC_threshhold"] = kp_AUC_threshold
            train_log[mode]["kp_metrics"]["results"]["Mean_pck"] = Mean_pck
            train_log[mode]["kp_metrics"]["results"]["Median_pck"] = Median_pck
            train_log[mode]["kp_metrics"]["results"]["std_pck"] = std_pck
            
            writer.add_scalar(f"{mode}/kp_AUC", round(kp_metrics["l2_error_auc"], 5), epoch)
            writer.add_scalar(f"{mode}/Mean_pck", round(kp_metrics["l2_error_mean_px"], 5), epoch)
            writer.add_scalar(f"{mode}/Median_pck", round(kp_metrics["l2_error_median_px"], 5), epoch)
            writer.add_scalar(f"{mode}/std_pck", round(kp_metrics["l2_error_std_px"], 5), epoch)
            
        else:
            train_log[mode]["kp_metrics"]["results"] = ["No in-frame gt keypoints were detected."]
    else:
        train_log[mode]["kp_metrics"]["correctness"].append("No in-frame gt keypoints.")
    
    n_pnp_possible = pnp_results["num_pnp_possible"]
    if n_pnp_possible > 0:
        n_pnp_successful = pnp_results["num_pnp_found"]
        n_pnp_fails = pnp_results["num_pnp_not_found"]
        fail = "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    )
        success = "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    )
        writer.add_scalar(f"{mode}/pnp_success", round(float(n_pnp_successful) / float(n_pnp_possible) * 100.0, 3), epoch)
        train_log[mode]["pnp_results"]["correctness"] += [fail, success]
        ADD = "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    )
        ADD_AUC = "   AUC: {:.5f}".format(pnp_results["add_auc"])
        ADD_AUC_threshold = "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"])
        Mean_ADD = "   Mean: {:.5f}".format(pnp_results["add_mean"])
        Median_ADD = "   Median: {:.5f}".format(pnp_results["add_median"])
        Std_ADD = "   Std Dev: {:.5f}".format(pnp_results["add_std"])
        train_log[mode]["pnp_results"]["results"] = {}
        train_log[mode]["pnp_results"]["results"]["ADD"] = ADD
        train_log[mode]["pnp_results"]["results"]["ADD_AUC"] = ADD_AUC
        train_log[mode]["pnp_results"]["results"]["ADD_AUC_threshold"] = ADD_AUC_threshold
        train_log[mode]["pnp_results"]["results"]["Mean_ADD"] = Mean_ADD
        train_log[mode]["pnp_results"]["results"]["Median_ADD"] = Median_ADD
        train_log[mode]["pnp_results"]["results"]["Std_ADD"] = Std_ADD
        writer.add_scalar(f"{mode}/ADD_AUC", round(pnp_results["add_auc"], 5), epoch)
        writer.add_scalar(f"{mode}/Mean_ADD", round(pnp_results["add_mean"], 5), epoch)
        writer.add_scalar(f"{mode}/Median_ADD", round(pnp_results["add_median"], 5), epoch)
        writer.add_scalar(f"{mode}/Std_ADD", round(pnp_results["add_std"], 5), epoch)
            
    
    

def main(opt):
    set_random_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    
    # set local rank
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device=torch.device("cuda",opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    
    tb_path = os.path.join(opt.save_dir, 'tb')
    ckpt_path = os.path.join(opt.save_dir, 'ckpt')
    results_path = os.path.join(opt.save_dir, 'results')
    
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        exists_or_mkdir(results_path)
        exists_or_mkdir(tb_path)
        exists_or_mkdir(ckpt_path)
    writer = SummaryWriter(tb_path)

    input_data_path = opt.dataset # 
    val_data_path = opt.val_dataset
    found_data = find_ndds_seq_data_in_dir(input_data_path, is_ct=opt.is_ct)
    if opt.add_dataset:
        add_data = find_ndds_seq_data_in_dir(opt.add_dataset, is_ct=opt.is_ct)
        print("length of original found_data", len(found_data))
        found_data += add_data
        print("length of current found_data", len(found_data))
    
    val_data = find_ndds_seq_data_in_dir(val_data_path,is_ct=opt.is_ct)
        
    network_input_resolution = (480, 480) # 
    network_output_resolution = (120, 120) # 
    input_width, input_height = network_input_resolution
    network_input_resolution_transpose = (input_height, input_width) # 
    opt = opts().update_dataset_info_and_set_heads_dream(opt, 7, network_input_resolution_transpose)
    keypoint_names = opts().get_keypoint_names(opt)
    print("keypoint_names", keypoint_names)
    image_normalization = {"mean" : (0.5, 0.5, 0.5), "stdev" : (0.5, 0.5, 0.5)}

    Dataset = CenterTrackSeqDataset(
    found_data, 
    opt.robot, 
    keypoint_names, 
    opt, 
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    include_ground_truth=True,
    include_belief_maps=True,
    seq_frame = 3
    ) 
    ValDataset = CenterTrackSeqDataset(
    val_data, 
    opt.robot, 
    keypoint_names, 
    opt, 
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    include_ground_truth=True,
    include_belief_maps=True,
    seq_frame = 3
    ) 

    num_gpus = torch.cuda.device_count()
    opt.max_iters = (opt.num_epochs * len(Dataset)) // opt.batch_size // num_gpus + 10
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)
        
    trainer = Trainer(opt, model, optimizer)
    print('opt.gpus', opt.gpus)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    train_sampler = DistributedSampler(Dataset)
    train_loader = torch.utils.data.DataLoader(
        Dataset, sampler=train_sampler, batch_size=opt.batch_size, 
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )
    val_sampler = DistributedSampler(ValDataset)
    val_loader = torch.utils.data.DataLoader(
        ValDataset, sampler=val_sampler, batch_size=opt.batch_size, 
        num_workers=opt.num_workers, pin_memory=True, drop_last=True 
        )
    
    for epoch in tqdm(range(start_epoch + 1, opt.num_epochs + 1)):
        trainer.train(epoch, train_loader, opt.device, writer, phase=opt.phase)
        this_path = os.path.join(ckpt_path, "model_{}.pth".format(epoch))
        save_model(this_path, epoch, model, optimizer)
        
        
#        # validation
        mean_valid_loss_per_batch, mean_valid_hm_loss_per_batch, mean_valid_reg_loss_per_batch = trainer.valid_epoch(val_loader, opt.device, phase=opt.phase)
        training_log = {}
        training_log["validation"] = {}
        training_log["validation"]["mean_valid_loss_all"] = mean_valid_loss_per_batch
        training_log["validation"]["mean_valid_loss_hm"] = mean_valid_hm_loss_per_batch
        training_log["validation"]["mean_valid_loss_reg"] = mean_valid_reg_loss_per_batch
        
        writer.add_scalar(f"validation/mean_valid_loss_all", mean_valid_loss_per_batch, epoch)
        writer.add_scalar(f"validation/mean_valid_loss_hm", mean_valid_hm_loss_per_batch, epoch)
        writer.add_scalar(f"validation/mean_valid_loss_reg", mean_valid_reg_loss_per_batch, epoch)

                

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)


















