# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import csv
import math
import os
from PIL import Image as PILImage
import pickle
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import sgtapose
from rf_tools.LM import *
from itertools import combinations
from scipy.special import comb
import random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_multiframe_pose_real(gt_kps_pos_lists,
    dt_kps_proj_lists,
    opt,
    keypoint_names,
    image_raw_resolution,
    output_dir,
    camera_K,
    multi_frame,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    is_real=False,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
    ):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    gpu_ids = opt.gpus
    num_of_all_frames = len(dt_kps_proj_lists)
    assert len(dt_kps_proj_lists) == len(gt_kps_pos_lists)
    
    dt_kps_proj_nps = np.array(dt_kps_proj_lists) # M x 7 x 2
    gt_kps_pos_nps = np.array(gt_kps_pos_lists) # M x 7 x 3

    N = 2500
    F = np.arange(num_of_all_frames)

    comb_nums = comb(num_of_all_frames, multi_frame)
    if comb_nums > N:
        pnp_index = []
        for i in range(N):
            random_index = random.sample(range(num_of_all_frames), multi_frame)
            # print("random_index", random_index)
            pnp_index.append(random_index)
    else:
        pnp_index = [list(combinations(F, multi_frame))]
    
    # m = multi_frame
    sample_dt_kps_proj_nps = dt_kps_proj_nps[pnp_index] # n x m x 7 x 2
    sample_gt_kps_pos_nps = gt_kps_pos_nps[pnp_index] # n x m x 7 x 3
    # sample_gt_kps_proj_nps = gt_kps_proj_nps[pnp_index]

    assert sample_dt_kps_proj_nps.shape[1] == multi_frame
    assert sample_gt_kps_pos_nps.shape[1] == multi_frame
    
    nums = sample_dt_kps_proj_nps.shape[0]
    all_n_inframe_projs_gt = []
    pnp_add = []
    poses_xyzxyzw = []
    add_and_index = []
    for idx in range(nums):
        this_dt_kps_proj, this_gt_kps_pos = sample_dt_kps_proj_nps[idx], sample_gt_kps_pos_nps[idx]
        # this_dt_kps_proj : m x 7 x 2
        # this_gt_kps_pos : m x 7 x 3
        # this_gt_kps_proj : m x 7 x 2

        n_inframe_projs_gt = multi_frame * 7

        this_dt_kps_proj = this_dt_kps_proj.reshape(-1, 2)
        this_gt_kps_pos = this_gt_kps_pos.reshape(-1, 3)
        idx_good_detections = np.where(this_dt_kps_proj > -999.0)
        idx_good_detections_rows = np.unique(idx_good_detections[0])
        kp_projs_est_pnp = this_dt_kps_proj[idx_good_detections_rows, :]
        kp_pos_gt_pnp = this_gt_kps_pos[idx_good_detections_rows, :]
        
        # n_inframe_projs_gt = len(idx_good_detections_rows)
        pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K)     
    

def solve_multiframe_pnp_real(
    gt_kps_pos_lists,
    dt_kps_proj_lists,
    opt,
    keypoint_names,
    image_raw_resolution,
    output_dir,
    camera_K,
    multi_frame,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    is_real=False,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
    ):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    gpu_ids = opt.gpus
    
    num_of_all_frames = len(dt_kps_proj_lists)
    # print('len(dt)', len(dt_kps_proj_lists))
    # print('len(gt)', len(gt_kps_pos_lists))
    assert len(dt_kps_proj_lists) == len(gt_kps_pos_lists)
    # assert len(dt_kps_proj_lists) == len(gt_kps_proj_lists)

    dt_kps_proj_nps = np.array(dt_kps_proj_lists) # M x 7 x 2
    gt_kps_pos_nps = np.array(gt_kps_pos_lists) # M x 7 x 3
    # gt_kps_proj_nps = np.array(gt_kps_proj_lists)

    N = 2500
    F = np.arange(num_of_all_frames)

    comb_nums = comb(num_of_all_frames, multi_frame)
    if comb_nums > N:
        pnp_index = []
        for i in range(N):
            random_index = random.sample(range(num_of_all_frames), multi_frame)
            # print("random_index", random_index)
            pnp_index.append(random_index)
    else:
        pnp_index = [list(combinations(F, multi_frame))]
    
    # m = multi_frame
    sample_dt_kps_proj_nps = dt_kps_proj_nps[pnp_index] # n x m x 7 x 2
    sample_gt_kps_pos_nps = gt_kps_pos_nps[pnp_index] # n x m x 7 x 3
    # sample_gt_kps_proj_nps = gt_kps_proj_nps[pnp_index]

    assert sample_dt_kps_proj_nps.shape[1] == multi_frame
    assert sample_gt_kps_pos_nps.shape[1] == multi_frame
    # assert sample_gt_kps_proj_nps.shape[1] == multi_frame

    nums = sample_dt_kps_proj_nps.shape[0]
    all_n_inframe_projs_gt = []
    pnp_add = []
    poses_xyzxyzw = []
    add_and_index = []
    for idx in range(nums):
        this_dt_kps_proj, this_gt_kps_pos = sample_dt_kps_proj_nps[idx], sample_gt_kps_pos_nps[idx]
        # this_dt_kps_proj : m x 7 x 2
        # this_gt_kps_pos : m x 7 x 3
        # this_gt_kps_proj : m x 7 x 2

        n_inframe_projs_gt = multi_frame * 7

        this_dt_kps_proj = this_dt_kps_proj.reshape(-1, 2)
        this_gt_kps_pos = this_gt_kps_pos.reshape(-1, 3)
        idx_good_detections = np.where(this_dt_kps_proj > -999.0)
        idx_good_detections_rows = np.unique(idx_good_detections[0])
        kp_projs_est_pnp = this_dt_kps_proj[idx_good_detections_rows, :]
        kp_pos_gt_pnp = this_gt_kps_pos[idx_good_detections_rows, :]
        
        # n_inframe_projs_gt = len(idx_good_detections_rows)
        pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K)
        # print('pnp_index', pnp_index)
        if pnp_retval:
            if opt.rf:
                # print("Introducing 3D refinement!!!")
                
                x,y,z,w = quaternion.tolist()
                # print('quaternion start', quaternion)
                quat_init = np.array([w,x,y,z]).reshape(1,4)
                trans_init = np.array(translation).reshape(1, 3)
                num_pt = kp_pos_gt_pnp.shape[0]
                x2d = kp_projs_est_pnp
                x2d_rep = []
                for x in kp_pos_gt_pnp:
                    #quat_init:(1,4),trans_init(1,3),quat_init[0]:(4,)
                    x2d_rep_i = camera_K @ (get_new_point_from_quaternion(x,quat_init[0]) + trans_init).T
                    x2d_rep_i[0]/=x2d_rep_i[2]
                    x2d_rep_i[1]/=x2d_rep_i[2]
                    x2d_rep.append(x2d_rep_i[0:2])
                x2d_rep = np.array(x2d_rep).squeeze()
                
                kp, _ = x2d_rep.shape
                distance_sq = np.linalg.norm((x2d-x2d_rep), axis=-1)**2
                # distances += distance_sq.tolist()
                distance_sq = distance_sq.reshape(kp, 1)
                distance_sq = np.repeat(distance_sq, 2, axis=-1)
                # print("dis", distance_sq)
                
                weights = get_weights(num_pt,distance_sq)
                start = time.perf_counter()
                # register_GN((7，2),(7,3), quat_init:(1,4),trans_init(1,3)
                # print("num_pt", num_pt)
                quat, T = register_GN_C(kp_projs_est_pnp,kp_pos_gt_pnp, quat_init, trans_init, weights, camera_K, num_pt)
                first = time.perf_counter()
                #quat:(4,),T:(3,)
                quat = torch.tensor(quat).view(1,4)
                T = torch.tensor(T).view(3,1)
                if torch.isnan(quat).any() or torch.isnan(T).any():
                    # print("quaternian isnan", quaternion)
                    x,y,z,w = quaternion.tolist()
                    quat = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                    T = translation
                    # ooo+=1
                    # timesum.append(first - start)


                poses_xyzxyzw.append(T.tolist() + quat.tolist()[0][1:] + quat.tolist()[0][:1])
                T = torch.tensor(T).view(3,1)
                quat = torch.tensor((quat)).view(1,4)
                #T:tensor(3,1),quat:tensor(1,4)
                add1 = sgtapose.geometric_vision.add_from_pose_tensor(
                    T, quat, kp_pos_gt_pnp, camera_K
                )
                translation = torch.tensor(translation).view(3, 1)
                # print("last quaternion", quaternion)
                x,y,z,w = quaternion.tolist()
                quaternion = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                add2 = sgtapose.geometric_vision.add_from_pose_tensor(
                    translation, quaternion, kp_pos_gt_pnp, camera_K
                )
                
                # print("refine add", add1)
                # print("original add", add2)
                    
                add = min(add1, add2)
            else:
                poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())
                add = sgtapose.geometric_vision.add_from_pose(
                    translation, quaternion, kp_pos_gt_pnp, camera_K
                )
            # print("idx", idx)
            # add_and_index.append([add, pnp_index[0][idx]])
        else:
            poses_xyzxyzw.append([-999.99] * 7)
            add = -999.99

        pnp_add.append(add)
        all_n_inframe_projs_gt.append(n_inframe_projs_gt)
    
    pnp_results = pnp_metrics(pnp_add, all_n_inframe_projs_gt)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def print_to_screen_and_file(file, text):
        print(text)
        file.write(text + "\n")
    results_log_path = os.path.join(output_dir, "analysis_results.txt")
    with open(results_log_path, "w") as f:
        if pnp_analysis:
            n_pnp_possible = pnp_results["num_pnp_possible"]
            if n_pnp_possible > 0:
                n_pnp_successful = pnp_results["num_pnp_found"]
                n_pnp_fails = pnp_results["num_pnp_not_found"]
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(pnp_results["add_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"]),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(pnp_results["add_mean"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(pnp_results["add_median"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(pnp_results["add_std"])
                )
            else:
                print_to_screen_and_file(f, "No frames where PNP is possible.")

            print_to_screen_and_file(f, "")
        
    # add_and_index.sort()
    # add_and_index_np = np.array(add_and_index)
    
    # with open('/root/autodl-tmp/yangtian/summer_ty/DREAM-master/dream_geo/ours.pk','wb') as file:
        # pickle.dump(add_and_index_np,file)
    
    return pnp_results




def solve_multiframe_pnp(
    json_lists,
    detected_kp_proj_lists,
    opt,
    keypoint_names,
    image_raw_resolution,
    output_dir,
    multiframe=2,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    is_real=False,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
    ):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    gpu_ids = opt.gpus
    dataset_path = "/root/autodl-tmp/dream_data/data/real"
    
    # Input argument handling
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), 'If specified, "batch_size" must be a positive integer.'
    assert (
        isinstance(num_workers, int) and num_workers >= 0
    ), 'If specified, "num_workers" must be an integer greater than or equal to zero.'

    all_gt_kp_projs = []
    all_dt_kp_projs = []
    all_gt_kp_pos = []
    all_kp_projs_blend = []
    all_json_list = []
    sample_results = []

    sample_results = []
    sample_idx = 0

    with torch.no_grad():
        print("Conducting inference...")
        for _, sample in enumerate(zip(json_lists, detected_kp_proj_lists)):
            this_json_list, this_detected_kp_proj_list = sample
            this_gt_kp_proj, this_gt_kp_pos, this_dt_kp_proj = [], [], []
            for this_json, this_detected_kp_proj in zip(this_json_list, this_detected_kp_proj_list):
                this_detected_kp_proj_np = np.array(this_detected_kp_proj)
                gt_kps_raw,  gt_kps_pos, dt_kps_raw = [], [], []
                parser = YAML(typ="safe")
                with open(this_json, "r") as f:
                    data = parser.load(f.read().replace('\t', ' '))
                # data = data[0]
                if is_real:
                    data = data["objects"][0]
                else:
                    data = data[0]
                object_keypoints = data["keypoints"]
                for idx, kp_name in enumerate(keypoint_names):
                    gt_kps_raw.append(object_keypoints[idx]["projected_location"])
                    if is_real:
                        gt_kps_pos.append(object_keypoints[idx]["location"])
                    else:
                        gt_kps_pos.append(object_keypoints[idx]["location_wrt_cam"])
                # gt_kps_pos.append(object_keypoints[idx]["location"])   
                gt_kps_raw = np.array(gt_kps_raw, dtype=np.float32)
                gt_kps_pos = np.array(gt_kps_pos, dtype=np.float32)

                this_dt_kp_proj.append(this_detected_kp_proj_np.tolist())
                this_gt_kp_proj.append(gt_kps_raw.tolist())
                this_gt_kp_pos.append(gt_kps_pos.tolist())

            
            
                # all_json_list.append(gt_jsons_list.tolist())
            all_gt_kp_projs.append(this_gt_kp_proj)
            all_dt_kp_projs.append(this_dt_kp_proj)
            all_gt_kp_pos.append(this_gt_kp_pos)
        
        assert len(all_gt_kp_projs) == len(all_dt_kp_projs)
        assert len(all_gt_kp_projs) == len(all_gt_kp_pos)

        # solve pnp in multiframes K
        all_gt_kp_projs = np.array(all_gt_kp_projs)
        all_gt_kp_pos = np.array(all_gt_kp_pos)
        all_dt_kp_projs = np.array(all_dt_kp_projs)
        pnp_attempts_successful = []
        poses_xyzxyzw = []
        all_n_inframe_projs_gt = []
        pnp_add = []
        distances = []
        
        if is_real:
            camera_data_path = os.path.join(dataset_path, is_real, "_camera_settings.json")
            camera_K = sgtapose.utilities.load_camera_intrinsics(camera_data_path)
        else:
            camera_K = np.array([[502.30, 0.0, 319.5], [0, 502.30, 179.5], [0, 0, 1]])

        for this_gt_kp_projs, this_gt_kp_pos, this_dt_kp_projs in zip(all_gt_kp_projs, \
                all_gt_kp_pos, all_dt_kp_projs):

            this_dt_kp_projs_np = np.array(this_dt_kp_projs)
            this_gt_kp_projs_np = np.array(this_gt_kp_projs)
            this_gt_kp_pos_np = np.array(this_gt_kp_pos)
            # n_inframe_projs_gt = 0
            for ind, ind_sample in enumerate(zip(
            this_dt_kp_projs, this_gt_kp_projs, this_gt_kp_pos
            )):
                # print('ind', ind)
                #if ind % multiframe != 0:
                if ind < multiframe - 1:
                    continue
                kp_projs_est, kp_projs_gt, kp_pos_gt = ind_sample
                n_inframe_projs_gt = 0
                for kp_proj_gt in kp_projs_gt:
                    if (
                        0.0 < kp_proj_gt[0]
                        and kp_proj_gt[0] < image_raw_resolution[0]
                        and 0.0 < kp_proj_gt[1]
                        and kp_proj_gt[1] < image_raw_resolution[1]
                        ):
                        n_inframe_projs_gt += 1
                
                all_n_inframe_projs_gt.append(n_inframe_projs_gt)
                # print("all_n_inframe_projs_gt", all_n_inframe_projs_gt)
                
                sample_info = {}
                sample_info["name"] = ind
                sample_results.append(sample_info)
                
                # print("shape", this_dt_kp_projs_np.shape)
                multi_kp_projs_est = this_dt_kp_projs_np[ind-multiframe+1:ind+1, :, :].reshape(-1, 2)
                multi_kp_projs_gt = this_gt_kp_projs_np[ind-multiframe+1:ind+1, :, :].reshape(-1, 2)
                multi_kp_pos_gt = this_gt_kp_pos_np[ind-multiframe+1:ind+1, :].reshape(-1, 3)

                idx_good_detections = np.where(multi_kp_projs_est > -999.0)
                idx_good_detections_rows = np.unique(idx_good_detections[0])
                kp_projs_est_pnp = multi_kp_projs_est[idx_good_detections_rows, :]
                kp_projs_gt_pnp = multi_kp_projs_gt[idx_good_detections_rows, :]
                kp_pos_gt_pnp = multi_kp_pos_gt[idx_good_detections_rows, :]

                pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K
                )
                pnp_attempts_successful.append(pnp_retval)

                if pnp_retval:
                    if opt.rf:
                        # print("Introducing 3D refinement!!!")
                        
                        x,y,z,w = quaternion.tolist()
                        # print('quaternion start', quaternion)
                        quat_init = np.array([w,x,y,z]).reshape(1,4)
                        trans_init = np.array(translation).reshape(1, 3)
                        num_pt = kp_pos_gt_pnp.shape[0]
                        x2d = kp_projs_est_pnp
                        x2d_rep = []
                        for x in kp_pos_gt_pnp:
                            #quat_init:(1,4),trans_init(1,3),quat_init[0]:(4,)
                            x2d_rep_i = camera_K @ (get_new_point_from_quaternion(x,quat_init[0]) + trans_init).T
                            x2d_rep_i[0]/=x2d_rep_i[2]
                            x2d_rep_i[1]/=x2d_rep_i[2]
                            x2d_rep.append(x2d_rep_i[0:2])
                        x2d_rep = np.array(x2d_rep).squeeze()
                        
                        kp, _ = x2d_rep.shape
                        distance_sq = np.linalg.norm((x2d-x2d_rep), axis=-1)**2
                        distances += distance_sq.tolist()
                        distance_sq = distance_sq.reshape(kp, 1)
                        distance_sq = np.repeat(distance_sq, 2, axis=-1)
                        # print("dis", distance_sq)
                        
                        weights = get_weights(num_pt,distance_sq)
                        start = time.perf_counter()
                        # register_GN((7，2),(7,3), quat_init:(1,4),trans_init(1,3)
                        # print("num_pt", num_pt)
                        quat, T = register_GN_C(kp_projs_est_pnp,kp_pos_gt_pnp, quat_init, trans_init, weights, camera_K, num_pt)
                        first = time.perf_counter()
                        #quat:(4,),T:(3,)
                        quat = torch.tensor(quat).view(1,4)
                        T = torch.tensor(T).view(3,1)
                        if torch.isnan(quat).any() or torch.isnan(T).any():
                            # print("quaternian isnan", quaternion)
                            x,y,z,w = quaternion.tolist()
                            quat = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                            T = translation
                            # ooo+=1
                            # timesum.append(first - start)


                        poses_xyzxyzw.append(T.tolist() + quat.tolist()[0][1:] + quat.tolist()[0][:1])
                        T = torch.tensor(T).view(3,1)
                        quat = torch.tensor((quat)).view(1,4)
                        #T:tensor(3,1),quat:tensor(1,4)
                        add1 = sgtapose.geometric_vision.add_from_pose_tensor(
                            T, quat, kp_pos_gt_pnp, camera_K
                        )
                        translation = torch.tensor(translation).view(3, 1)
                        # print("last quaternion", quaternion)
                        x,y,z,w = quaternion.tolist()
                        quaternion = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                        add2 = sgtapose.geometric_vision.add_from_pose_tensor(
                            translation, quaternion, kp_pos_gt_pnp, camera_K
                        )
                        
                        # print("refine add", add1)
                        # print("original add", add2)
                            
                        add = min(add1, add2)
                    else:
                        poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())
                        add = sgtapose.geometric_vision.add_from_pose(
                            translation, quaternion, kp_pos_gt_pnp, camera_K
                        )
                else:
                    poses_xyzxyzw.append([-999.99] * 7)
                    add = -999.99

                pnp_add.append(add)

        pnp_path = os.path.join(output_dir, f"{opt.is_real}_{multiframe}_pnp_results.csv")
        sample_names = [x["name"] for x in sample_results]
        write_pnp_csv(
            pnp_path,
            sample_names,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
        pnp_results = pnp_metrics(pnp_add, all_n_inframe_projs_gt)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def print_to_screen_and_file(file, text):
        print(text)
        file.write(text + "\n")
    results_log_path = os.path.join(output_dir, "analysis_results.txt")
    with open(results_log_path, "w") as f:
        if pnp_analysis:
            n_pnp_possible = pnp_results["num_pnp_possible"]
            if n_pnp_possible > 0:
                n_pnp_successful = pnp_results["num_pnp_found"]
                n_pnp_fails = pnp_results["num_pnp_not_found"]
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(pnp_results["add_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"]),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(pnp_results["add_mean"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(pnp_results["add_median"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(pnp_results["add_std"])
                )
            else:
                print_to_screen_and_file(f, "No frames where PNP is possible.")

            print_to_screen_and_file(f, "")
    return pnp_results

def analyze_ndds_center_dream_dataset(
    json_list, 
    detected_kp_proj_list,
    opt, 
    keypoint_names,
    image_raw_resolution,
    output_dir,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    is_real=False,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
    set_mode=None,
    dataset_path=None,
):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    gpu_ids = opt.gpus
    if dataset_path is None:
        dataset_path = "/DATA/disk1/hyperplane/ty_data/"
    
    
    # Input argument handling
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), 'If specified, "batch_size" must be a positive integer.'
    assert (
        isinstance(num_workers, int) and num_workers >= 0
    ), 'If specified, "num_workers" must be an integer greater than or equal to zero.'


    all_kp_projs_gt_raw = []
    all_kp_projs_detected_raw = []
    all_gt_kp_positions = []
    all_dream_7_gt_kp_positions = []
    all_kp_projs_blend = []
    all_json_list = []

    sample_results = []
    sample_idx = 0
    
    if is_real:
        camera_data_path = os.path.join(dataset_path, is_real, "_camera_settings.json")
        camera_K = sgtapose.utilities.load_camera_intrinsics(camera_data_path)
    else:
        camera_K = np.array([[502.30, 0.0, 319.75], [0, 502.30, 179.75], [0, 0, 1]])
    
    with torch.no_grad():
        print("Conducting inference...")
        for idx, sample in enumerate(zip(json_list, detected_kp_proj_list)):
            this_json_path, this_detected_kps_raw = sample
            this_detected_kps_raw = np.array(this_detected_kps_raw)
            
            # 
            gt_kps_raw,  gt_kps_pos, gt_kps_blend, gt_jsons_list = [], [], [], []
            parser = YAML(typ="safe")
            with open(this_json_path, "r") as f:
                data = parser.load(f.read().replace('\t', ' '))
            # data = data[0]
            if is_real and "panda" in is_real:
                data = data["objects"][0]
            else:
                data = data[0]
            object_keypoints = data["keypoints"]
            
            if is_real and "panda" in is_real:
                #print(object_keypoints)
                #print(keypoint_names)
                for idx, kp_name in enumerate(keypoint_names):
                    # print("object_keypoints", object_keypoints[idx])
                    projections = camera_K @ np.array(object_keypoints[idx]["location"])
                    projections /= projections[2]
                    gt_kps_raw.append(projections.tolist()[:2])
                    gt_kps_pos.append(object_keypoints[idx]["location"])
                    gt_jsons_list.append(os.path.join(this_json_path, str(idx)))
            else:
                count = 0
                for idx, kp_name in enumerate(keypoint_names):
                    while object_keypoints[count]["Name"] != kp_name:
                        count += 1
                    assert (object_keypoints[count]["Name"] == kp_name)
                    projections = np.array(object_keypoints[count]["location_wrt_cam"])
                    projections = camera_K @ projections
                    projections /= projections[2]
                    gt_kps_raw.append(projections.tolist()[:2])
                    gt_kps_pos.append(object_keypoints[count]["location_wrt_cam"])
                    gt_jsons_list.append(os.path.join(this_json_path, str(count)))
                
                
            
            
                # gt_kps_pos.append(object_keypoints[idx]["location"])
            gt_kps_raw = np.array(gt_kps_raw, dtype=np.float32)
            gt_kps_pos = np.array(gt_kps_pos, dtype=np.float32)
            gt_jsons_list = np.array(gt_jsons_list)
            
            all_kp_projs_detected_raw.append(this_detected_kps_raw.tolist())
            all_kp_projs_gt_raw.append(gt_kps_raw.tolist())
            all_json_list.append(gt_jsons_list.tolist())
                            
            
            if pnp_analysis:
                all_gt_kp_positions.append(gt_kps_pos.tolist())

            # Metric is just L2 error at the raw image frame for in-frame keypoints if network detects one (original, before network input)
            kp_l2_err = []
            for this_kp_detect_raw, this_kp_gt_raw in zip(
                this_detected_kps_raw, gt_kps_raw
            ):  
                gt_kps_blend.append(this_kp_gt_raw.tolist()+this_kp_detect_raw.tolist())
                if (
                    (
                        this_kp_detect_raw[0] <= -999.0
                        and this_kp_detect_raw[1] <= -999.0
                    )
                    or this_kp_gt_raw[0] < 0
                    or this_kp_gt_raw[0] > image_raw_resolution[0] 
                    or this_kp_gt_raw[1] < 0.0
                    or this_kp_gt_raw[1] > image_raw_resolution[1]
                ):
                    continue
                
                
                kp_l2_err.append(
                    np.linalg.norm(this_kp_detect_raw - this_kp_gt_raw)
                )

            if kp_l2_err:
                this_metric = np.mean(kp_l2_err)
            else:
                this_metric = 999.999

            this_sample_info = {
                "json_paths": this_json_path,
                "name": this_json_path.split("/")[-2]
            }
            sample_results.append((sample_idx, this_sample_info, this_metric))
            all_kp_projs_blend.append(gt_kps_blend)
            # print(gt_kps_blend)

            sample_idx += 1

    all_kp_projs_detected_raw = np.array(all_kp_projs_detected_raw)
    all_kp_projs_gt_raw = np.array(all_kp_projs_gt_raw)
    all_json_np = np.array(all_json_list)
    
    # print(np.array(all_kp_projs_blend).shape)
    all_kp_projs_blend_np = np.array(all_kp_projs_blend).reshape(-1, 4)
    np.savetxt(os.path.join(output_dir, "gt_and_detected.txt"), all_kp_projs_blend_np)
    

    # Write keypoint file
    syn = not opt.is_ct
    print('syn', syn)
    n_samples = len(sample_results)
    kp_metrics = keypoint_metrics(
        all_kp_projs_detected_raw.reshape(n_samples * opt.num_classes, 2),
        all_kp_projs_gt_raw.reshape(n_samples * opt.num_classes, 2),
        all_json_np.reshape(n_samples * opt.num_classes, -1),
        image_raw_resolution,
        syn=syn
    )
    keypoint_path = os.path.join(output_dir, f"{is_real}_keypoints.csv")
    if not opt.is_real:
        keypoint_path = os.path.join(output_dir, f"{set_mode}_keypoints.csv")
    sample_names = [x[1]["name"] for x in sample_results]

    write_keypoint_csv(
        keypoint_path, sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw
    )

    # PNP analysis
    pnp_attempts_successful = []
    poses_xyzxyzw = []
    all_n_inframe_projs_gt = []
    pnp_add = []
    distances = []

    if pnp_analysis:
        all_gt_kp_positions = np.array(all_gt_kp_positions)
        # camera_K = np.array([[502.30, 0.0, 319.5], [0, 502.30, 179.5], [0, 0, 1]])
        if is_real:
            camera_data_path = os.path.join(dataset_path, is_real, "_camera_settings.json")
            camera_K = sgtapose.utilities.load_camera_intrinsics(camera_data_path)
        else:
            camera_K = np.array([[502.30, 0.0, 319.75], [0, 502.30, 179.75], [0, 0, 1]])
            # camera_K = np.array([[1004.6, 0.0, 639.5], [0.0, 1004.6, 359.5], [0.0, 0.0, 1.0]])
        for kp_projs_est, kp_projs_gt, kp_pos_gt in zip(
            all_kp_projs_detected_raw, all_kp_projs_gt_raw, all_gt_kp_positions
        ):

            n_inframe_projs_gt = 0
            for kp_proj_gt in kp_projs_gt:
                if (
                    0.0 < kp_proj_gt[0]
                    and kp_proj_gt[0] < image_raw_resolution[0]
                    and 0.0 < kp_proj_gt[1]
                    and kp_proj_gt[1] < image_raw_resolution[1]
                ):
                    n_inframe_projs_gt += 1

            idx_good_detections = np.where(kp_projs_est > -999.0)
            idx_good_detections_rows = np.unique(idx_good_detections[0])
            kp_projs_est_pnp = kp_projs_est[idx_good_detections_rows, :]
            kp_projs_gt_pnp = kp_projs_gt[idx_good_detections_rows, :]
            kp_pos_gt_pnp = kp_pos_gt[idx_good_detections_rows, :]
            
            #print("camera_K", camera_K)
            pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K
            )
            
#            pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
#                kp_pos_gt_pnp, kp_projs_gt_pnp, camera_K
#            )
#            pnp_retval, translation, quaternion, inliers = sgtapose.geometric_vision.solve_pnp_ransac(kp_pos_gt_pnp, kp_projs_est_pnp, camera_K)
            
            #print("pnp_retval", pnp_retval)
            pnp_attempts_successful.append(pnp_retval)

            all_n_inframe_projs_gt.append(n_inframe_projs_gt)

            if pnp_retval:
                if opt.rf:
                    # print("Introducing 3D refinement!!!")
                    
                    x,y,z,w = quaternion.tolist()
                    # print('quaternion start', quaternion)
                    quat_init = np.array([w,x,y,z]).reshape(1,4)
                    trans_init = np.array(translation).reshape(1, 3)
                    num_pt = kp_pos_gt_pnp.shape[0]
                    x2d = kp_projs_est_pnp
                    x2d_rep = []
                    for x in kp_pos_gt_pnp:
                        #quat_init:(1,4),trans_init(1,3),quat_init[0]:(4,)
                        x2d_rep_i = camera_K @ (get_new_point_from_quaternion(x,quat_init[0]) + trans_init).T
                        x2d_rep_i[0]/=x2d_rep_i[2]
                        x2d_rep_i[1]/=x2d_rep_i[2]
                        x2d_rep.append(x2d_rep_i[0:2])
                    x2d_rep = np.array(x2d_rep).squeeze()
                    
                    kp, _ = x2d_rep.shape
                    distance_sq = np.linalg.norm((x2d-x2d_rep), axis=-1)**2
                    distances += distance_sq.tolist()
                    distance_sq = distance_sq.reshape(kp, 1)
                    distance_sq = np.repeat(distance_sq, 2, axis=-1)
                    # print("dis", distance_sq)
                    
                    weights = get_weights(num_pt,distance_sq)
                    start = time.perf_counter()
                    # register_GN((7，2),(7,3), quat_init:(1,4),trans_init(1,3)
                    # print("num_pt", num_pt)
                    quat, T = register_GN_C(kp_projs_est_pnp,kp_pos_gt_pnp, quat_init, trans_init, weights, camera_K, num_pt)
                    first = time.perf_counter()
                    #quat:(4,),T:(3,)
                    quat = torch.tensor(quat).view(1,4)
                    T = torch.tensor(T).view(3,1)
                    if torch.isnan(quat).any() or torch.isnan(T).any():
                        # print("quaternian isnan", quaternion)
                        x,y,z,w = quaternion.tolist()
                        quat = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                        T = translation
                        # ooo+=1
                        # timesum.append(first - start)


                    poses_xyzxyzw.append(T.tolist() + quat.tolist()[0][1:] + quat.tolist()[0][:1])
                    T = torch.tensor(T).view(3,1)
                    quat = torch.tensor((quat)).view(1,4)
                    #T:tensor(3,1),quat:tensor(1,4)
                    add1 = sgtapose.geometric_vision.add_from_pose_tensor(
                        T, quat, kp_pos_gt_pnp, camera_K
                    )
                    translation = torch.tensor(translation).view(3, 1)
                    # print("last quaternion", quaternion)
                    x,y,z,w = quaternion.tolist()
                    quaternion = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                    add2 = sgtapose.geometric_vision.add_from_pose_tensor(
                        translation, quaternion, kp_pos_gt_pnp, camera_K
                    )
                    
                    print("refine add", add1)
                    print("original add", add2)
                        
                    add = min(add1, add2)
                else:
                    poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())
                    add = sgtapose.geometric_vision.add_from_pose(
                        translation, quaternion, kp_pos_gt_pnp, camera_K
                    )
            else:
                poses_xyzxyzw.append([-999.99] * 7)
                add = -999.99

            pnp_add.append(add)
            #print(add)
        
        distances_np = np.array(distances)
        # np.savetxt("/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/dis.txt", distances_np)

        pnp_path = os.path.join(output_dir, f"{is_real}_pnp_results.csv")
        if not opt.is_real:
            pnp_path = os.path.join(output_dir, f"{set_mode}_pnp_results.csv")
        write_pnp_csv(
            pnp_path,
            sample_names,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
        # print("pnp_add", pnp_add)
        pnp_results = pnp_metrics(pnp_add, all_n_inframe_projs_gt)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def print_to_screen_and_file(file, text):
        print(text)
        file.write(text + "\n")
    if is_real:
        results_log_path = os.path.join(output_dir, is_real + "_analysis_results.txt")
    else:
        results_log_path = os.path.join(output_dir, "_analysis_results.txt")
    with open(results_log_path, "w") as f:

        # Write results header
        print_to_screen_and_file(
            f, "Analysis results for dataset: {}".format('Seq')
        )
        print_to_screen_and_file(
            f, "Number of frames in this dataset: {}".format(n_samples)
        )
        print_to_screen_and_file(
            f, "Using network config defined from: {}".format(None)
        )
        print_to_screen_and_file(f, "")

        # Write keypoint metric summary to file
        if kp_metrics["num_gt_outframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints not found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
        else:
            print_to_screen_and_file(f, "No out-of-frame gt keypoints.")

        if kp_metrics["num_gt_inframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints not found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            if kp_metrics["num_found_gt_inframe"] > 0:
                print_to_screen_and_file(
                    f,
                    "L2 error (px) for in-frame keypoints (n = {}):".format(
                        kp_metrics["num_found_gt_inframe"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(kp_metrics["l2_error_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(
                        kp_metrics["l2_error_auc_thresh_px"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(kp_metrics["l2_error_mean_px"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(kp_metrics["l2_error_median_px"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(kp_metrics["l2_error_std_px"])
                )
            else:
                print_to_screen_and_file(f, "No in-frame gt keypoints were detected.")
        else:
            print_to_screen_and_file(f, "No in-frame gt keypoints.")

        print_to_screen_and_file(f, "")

        if pnp_analysis:
            n_pnp_possible = pnp_results["num_pnp_possible"]
            if n_pnp_possible > 0:
                n_pnp_successful = pnp_results["num_pnp_found"]
                n_pnp_fails = pnp_results["num_pnp_not_found"]
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(pnp_results["add_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"]),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(pnp_results["add_mean"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(pnp_results["add_median"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(pnp_results["add_std"])
                )
            else:
                print_to_screen_and_file(f, "No frames where PNP is possible.")

            print_to_screen_and_file(f, "")

        
    if pnp_analysis:
        return (
            kp_metrics, 
            pnp_results,
            sample_names,
            all_kp_projs_detected_raw,
            all_kp_projs_gt_raw,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
    else:
        return kp_metrics, pnp_results, sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw
        
def analyze_ndds_center_dream_ours_42_dataset(
    json_list, 
    detected_kp_proj_list,
    opt, 
    keypoint_names,
    image_raw_resolution,
    output_dir,
    visualize_belief_maps=True,
    pnp_analysis=True,
    force_overwrite=False,
    is_real=False,
    batch_size=16,
    num_workers=8,
    gpu_ids=None,
    set_mode=None,
    dataset_path=None,
):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    gpu_ids = opt.gpus
    if dataset_path is None:
        dataset_path = "/DATA/disk1/hyperplane/ty_data/"
    
    
    # Input argument handling
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), 'If specified, "batch_size" must be a positive integer.'
    assert (
        isinstance(num_workers, int) and num_workers >= 0
    ), 'If specified, "num_workers" must be an integer greater than or equal to zero.'


    all_kp_projs_gt_raw = []
    all_kp_projs_detected_raw = []
    all_gt_kp_positions = []
    all_dream_7_gt_kp_positions = []
    all_kp_projs_blend = []
    all_json_list = []

    sample_results = []
    sample_idx = 0
    
    if is_real:
        camera_data_path = os.path.join(dataset_path, is_real, "_camera_settings.json")
        camera_K = sgtapose.utilities.load_camera_intrinsics(camera_data_path)
    else:
        camera_K = np.array([[502.30, 0.0, 319.75], [0, 502.30, 179.75], [0, 0, 1]])
    
    with torch.no_grad():
        print("Conducting inference...")
        for idx, sample in enumerate(zip(json_list, detected_kp_proj_list)):
            this_json_path, this_detected_kps_raw = sample
            this_detected_kps_raw = np.array(this_detected_kps_raw)
            
            # 
            gt_kps_raw,  gt_kps_pos, gt_kps_blend, gt_jsons_list = [], [], [], []
            parser = YAML(typ="safe")
            with open(this_json_path, "r") as f:
                data = parser.load(f.read().replace('\t', ' '))
            # data = data[0]
            if is_real and "panda" in is_real:
                data = data["objects"][0]
            else:
                data = data[0]
            object_keypoints = data["keypoints"]
            object_joints = data["joints_3n_fixed_42"]
            
#            if is_real and "panda" in is_real:
#                #print(object_keypoints)
#                #print(keypoint_names)
#                for idx, kp_name in enumerate(keypoint_names):
#                    # print("object_keypoints", object_keypoints[idx])
#                    projections = camera_K @ np.array(object_keypoints[idx]["location"])
#                    projections /= projections[2]
#                    gt_kps_raw.append(projections.tolist()[:2])
#                    gt_kps_pos.append(object_keypoints[idx]["location"])
#                    gt_jsons_list.append(os.path.join(this_json_path, str(idx)))
#            else:
#                count = 0
#                for idx, kp_name in enumerate(keypoint_names):
#                    while object_keypoints[count]["Name"] != kp_name:
#                        count += 1
#                    assert (object_keypoints[count]["Name"] == kp_name)
#                    projections = np.array(object_keypoints[count]["location_wrt_cam"])
#                    projections = camera_K @ projections
#                    projections /= projections[2]
#                    gt_kps_raw.append(projections.tolist()[:2])
#                    gt_kps_pos.append(object_keypoints[count]["location_wrt_cam"])
#                    gt_jsons_list.append(os.path.join(this_json_path, str(count)))
            gt_kps_pos = np.array([i["location_wrt_cam"] for i in object_joints])
            gt_jsons_list = np.array([os.path.join(this_json_path, str(i)) for i in range(gt_kps_pos.shape[0])])
            gt_kps_raw = (camera_K @ gt_kps_pos.T).T
            gt_kps_raw[:, :2] /= gt_kps_raw[:, 2:]
            gt_kps_raw = gt_kps_raw[:, :2]
            gt_kps_dream_7_pos = np.array([i["location_wrt_cam"] for i in object_keypoints])[[0,2,3,4,6,7,8]]
            
            
            all_kp_projs_detected_raw.append(this_detected_kps_raw.tolist())
            all_kp_projs_gt_raw.append(gt_kps_raw.tolist())
            all_json_list.append(gt_jsons_list.tolist())
            all_dream_7_gt_kp_positions.append(gt_kps_dream_7_pos.tolist())
                            
            
            if pnp_analysis:
                all_gt_kp_positions.append(gt_kps_pos.tolist())

            # Metric is just L2 error at the raw image frame for in-frame keypoints if network detects one (original, before network input)
            kp_l2_err = []
            for this_kp_detect_raw, this_kp_gt_raw in zip(
                this_detected_kps_raw, gt_kps_raw
            ):  
                gt_kps_blend.append(this_kp_gt_raw.tolist()+this_kp_detect_raw.tolist())
                if (
                    (
                        this_kp_detect_raw[0] <= -999.0
                        and this_kp_detect_raw[1] <= -999.0
                    )
                    or this_kp_gt_raw[0] < 0
                    or this_kp_gt_raw[0] > image_raw_resolution[0] 
                    or this_kp_gt_raw[1] < 0.0
                    or this_kp_gt_raw[1] > image_raw_resolution[1]
                ):
                    continue
                
                
                kp_l2_err.append(
                    np.linalg.norm(this_kp_detect_raw - this_kp_gt_raw)
                )

            if kp_l2_err:
                this_metric = np.mean(kp_l2_err)
            else:
                this_metric = 999.999

            this_sample_info = {
                "json_paths": this_json_path,
                "name": this_json_path.split("/")[-2]
            }
            sample_results.append((sample_idx, this_sample_info, this_metric))
            all_kp_projs_blend.append(gt_kps_blend)
            # print(gt_kps_blend)

            sample_idx += 1

    all_kp_projs_detected_raw = np.array(all_kp_projs_detected_raw)
    all_kp_projs_gt_raw = np.array(all_kp_projs_gt_raw)
    all_json_np = np.array(all_json_list)
    
    # print(np.array(all_kp_projs_blend).shape)
    all_kp_projs_blend_np = np.array(all_kp_projs_blend).reshape(-1, 4)
    np.savetxt(os.path.join(output_dir, "gt_and_detected.txt"), all_kp_projs_blend_np)
    

    # Write keypoint file
    syn = not opt.is_ct
    print('syn', syn)
    n_samples = len(sample_results)
    kp_metrics = keypoint_metrics(
        all_kp_projs_detected_raw.reshape(n_samples * opt.num_classes, 2),
        all_kp_projs_gt_raw.reshape(n_samples * opt.num_classes, 2),
        all_json_np.reshape(n_samples * opt.num_classes, -1),
        image_raw_resolution,
        syn=syn
    )
    keypoint_path = os.path.join(output_dir, f"{is_real}_keypoints.csv")
    if not opt.is_real:
        keypoint_path = os.path.join(output_dir, f"{set_mode}_keypoints.csv")
    sample_names = [x[1]["name"] for x in sample_results]

    write_keypoint_csv(
        keypoint_path, sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw
    )

    # PNP analysis
    pnp_attempts_successful = []
    poses_xyzxyzw = []
    all_n_inframe_projs_gt = []
    pnp_add = []
    distances = []

    if pnp_analysis:
        all_gt_kp_positions = np.array(all_gt_kp_positions)
        # camera_K = np.array([[502.30, 0.0, 319.5], [0, 502.30, 179.5], [0, 0, 1]])
        if is_real:
            camera_data_path = os.path.join(dataset_path, is_real, "_camera_settings.json")
            camera_K = sgtapose.utilities.load_camera_intrinsics(camera_data_path)
        else:
            camera_K = np.array([[502.30, 0.0, 319.75], [0, 502.30, 179.75], [0, 0, 1]])
            # camera_K = np.array([[1004.6, 0.0, 639.5], [0.0, 1004.6, 359.5], [0.0, 0.0, 1.0]])
        for kp_projs_est, kp_projs_gt, kp_pos_gt, dream_7_kp_pos_gt in zip(
            all_kp_projs_detected_raw, all_kp_projs_gt_raw, all_gt_kp_positions, all_dream_7_gt_kp_positions
        ):

            n_inframe_projs_gt = 0
            for kp_proj_gt in kp_projs_gt:
                if (
                    0.0 < kp_proj_gt[0]
                    and kp_proj_gt[0] < image_raw_resolution[0]
                    and 0.0 < kp_proj_gt[1]
                    and kp_proj_gt[1] < image_raw_resolution[1]
                ):
                    n_inframe_projs_gt += 1

            idx_good_detections = np.where(kp_projs_est > -999.0)
            idx_good_detections_rows = np.unique(idx_good_detections[0])
            kp_projs_est_pnp = kp_projs_est[idx_good_detections_rows, :]
            kp_projs_gt_pnp = kp_projs_gt[idx_good_detections_rows, :]
            kp_pos_gt_pnp = kp_pos_gt[idx_good_detections_rows, :]
            dream_7_kp_pos_gt_pnp = dream_7_kp_pos_gt
            
            #print("camera_K", camera_K)
            pnp_retval, translation, quaternion = sgtapose.geometric_vision.solve_pnp(
                kp_pos_gt_pnp, kp_projs_est_pnp, camera_K
            )
            
            pnp_attempts_successful.append(pnp_retval)

            all_n_inframe_projs_gt.append(n_inframe_projs_gt)

            if pnp_retval:
                if opt.rf:
                    # print("Introducing 3D refinement!!!")
                    
                    x,y,z,w = quaternion.tolist()
                    # print('quaternion start', quaternion)
                    quat_init = np.array([w,x,y,z]).reshape(1,4)
                    trans_init = np.array(translation).reshape(1, 3)
                    num_pt = kp_pos_gt_pnp.shape[0]
                    x2d = kp_projs_est_pnp
                    x2d_rep = []
                    for x in kp_pos_gt_pnp:
                        #quat_init:(1,4),trans_init(1,3),quat_init[0]:(4,)
                        x2d_rep_i = camera_K @ (get_new_point_from_quaternion(x,quat_init[0]) + trans_init).T
                        x2d_rep_i[0]/=x2d_rep_i[2]
                        x2d_rep_i[1]/=x2d_rep_i[2]
                        x2d_rep.append(x2d_rep_i[0:2])
                    x2d_rep = np.array(x2d_rep).squeeze()
                    
                    kp, _ = x2d_rep.shape
                    distance_sq = np.linalg.norm((x2d-x2d_rep), axis=-1)**2
                    distances += distance_sq.tolist()
                    distance_sq = distance_sq.reshape(kp, 1)
                    distance_sq = np.repeat(distance_sq, 2, axis=-1)
                    # print("dis", distance_sq)
                    
                    weights = get_weights(num_pt,distance_sq)
                    start = time.perf_counter()
                    # register_GN((7，2),(7,3), quat_init:(1,4),trans_init(1,3)
                    # print("num_pt", num_pt)
                    quat, T = register_GN_C(kp_projs_est_pnp,kp_pos_gt_pnp, quat_init, trans_init, weights, camera_K, num_pt)
                    first = time.perf_counter()
                    #quat:(4,),T:(3,)
                    quat = torch.tensor(quat).view(1,4)
                    T = torch.tensor(T).view(3,1)
                    if torch.isnan(quat).any() or torch.isnan(T).any():
                        # print("quaternian isnan", quaternion)
                        x,y,z,w = quaternion.tolist()
                        quat = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                        T = translation
                        # ooo+=1
                        # timesum.append(first - start)


                    poses_xyzxyzw.append(T.tolist() + quat.tolist()[0][1:] + quat.tolist()[0][:1])
                    T = torch.tensor(T).view(3,1)
                    quat = torch.tensor((quat)).view(1,4)
                    #T:tensor(3,1),quat:tensor(1,4)
                    add1 = sgtapose.geometric_vision.add_from_pose_tensor(
                        T, quat, kp_pos_gt_pnp, camera_K
                    )
                    translation = torch.tensor(translation).view(3, 1)
                    # print("last quaternion", quaternion)
                    x,y,z,w = quaternion.tolist()
                    quaternion = torch.from_numpy(np.array([w,x,y,z])).view(1, 4)
                    add2 = sgtapose.geometric_vision.add_from_pose_tensor(
                        translation, quaternion, dream_7_kp_pos_gt_pnp, camera_K
                    )
                    
                    print("refine add", add1)
                    print("original add", add2)
                        
                    add = min(add1, add2)
                else:
                    poses_xyzxyzw.append(translation.tolist() + quaternion.tolist())
                    add = sgtapose.geometric_vision.add_from_pose(
                        translation, quaternion, dream_7_kp_pos_gt_pnp, camera_K
                    )
            else:
                poses_xyzxyzw.append([-999.99] * 7)
                add = -999.99

            pnp_add.append(add)
            #print(add)
        
        distances_np = np.array(distances)
        # np.savetxt("/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/dis.txt", distances_np)

        pnp_path = os.path.join(output_dir, f"{is_real}_pnp_results.csv")
        if not opt.is_real:
            pnp_path = os.path.join(output_dir, f"{set_mode}_pnp_results.csv")
        write_pnp_csv(
            pnp_path,
            sample_names,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
        # print("pnp_add", pnp_add)
        pnp_results = pnp_metrics(pnp_add, all_n_inframe_projs_gt)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def print_to_screen_and_file(file, text):
        print(text)
        file.write(text + "\n")
    if is_real:
        results_log_path = os.path.join(output_dir, is_real + "_analysis_results.txt")
    else:
        results_log_path = os.path.join(output_dir, "_analysis_results.txt")
    with open(results_log_path, "w") as f:

        # Write results header
        print_to_screen_and_file(
            f, "Analysis results for dataset: {}".format('Seq')
        )
        print_to_screen_and_file(
            f, "Number of frames in this dataset: {}".format(n_samples)
        )
        print_to_screen_and_file(
            f, "Using network config defined from: {}".format(None)
        )
        print_to_screen_and_file(f, "")

        # Write keypoint metric summary to file
        if kp_metrics["num_gt_outframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints not found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage out-of-frame gt keypoints found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_outframe"])
                    / float(kp_metrics["num_gt_outframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_outframe"],
                    kp_metrics["num_gt_outframe"],
                ),
            )
        else:
            print_to_screen_and_file(f, "No out-of-frame gt keypoints.")

        if kp_metrics["num_gt_inframe"] > 0:
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints not found (incorrect): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_missing_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_missing_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            print_to_screen_and_file(
                f,
                "Percentage in-frame gt keypoints found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                ),
            )
            if kp_metrics["num_found_gt_inframe"] > 0:
                print_to_screen_and_file(
                    f,
                    "L2 error (px) for in-frame keypoints (n = {}):".format(
                        kp_metrics["num_found_gt_inframe"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(kp_metrics["l2_error_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(
                        kp_metrics["l2_error_auc_thresh_px"]
                    ),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(kp_metrics["l2_error_mean_px"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(kp_metrics["l2_error_median_px"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(kp_metrics["l2_error_std_px"])
                )
            else:
                print_to_screen_and_file(f, "No in-frame gt keypoints were detected.")
        else:
            print_to_screen_and_file(f, "No in-frame gt keypoints.")

        print_to_screen_and_file(f, "")

        if pnp_analysis:
            n_pnp_possible = pnp_results["num_pnp_possible"]
            if n_pnp_possible > 0:
                n_pnp_successful = pnp_results["num_pnp_found"]
                n_pnp_fails = pnp_results["num_pnp_not_found"]
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    ),
                )
                print_to_screen_and_file(
                    f,
                    "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    ),
                )
                print_to_screen_and_file(
                    f, "   AUC: {:.5f}".format(pnp_results["add_auc"])
                )
                print_to_screen_and_file(
                    f,
                    "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"]),
                )
                print_to_screen_and_file(
                    f, "   Mean: {:.5f}".format(pnp_results["add_mean"])
                )
                print_to_screen_and_file(
                    f, "   Median: {:.5f}".format(pnp_results["add_median"])
                )
                print_to_screen_and_file(
                    f, "   Std Dev: {:.5f}".format(pnp_results["add_std"])
                )
            else:
                print_to_screen_and_file(f, "No frames where PNP is possible.")

            print_to_screen_and_file(f, "")

        
    if pnp_analysis:
        return (
            kp_metrics, 
            pnp_results,
            sample_names,
            all_kp_projs_detected_raw,
            all_kp_projs_gt_raw,
            pnp_attempts_successful,
            poses_xyzxyzw,
            pnp_add,
            all_n_inframe_projs_gt,
        )
    else:
        return kp_metrics, pnp_results, sample_names, all_kp_projs_detected_raw, all_kp_projs_gt_raw

def write_keypoint_csv(keypoint_path, sample_names, keypoints_detected, keypoints_gt):

    assert (
        keypoints_detected.shape == keypoints_gt.shape
    ), 'Expected "keypoints_detected" and "keypoints_gt" to have the same shape.'

    n_samples = len(sample_names)

    assert (
        n_samples == keypoints_detected.shape[0]
    ), "Expected number of sample names to equal the number of keypoint entries."

    n_keypoints = keypoints_detected.shape[1]
    n_keypoint_dims = keypoints_detected.shape[2]

    assert n_keypoint_dims == 2, "Expected the number of keypoint dimensions to be 2."

    n_keypoint_elements = n_keypoints * n_keypoint_dims

    with open(keypoint_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)

        kp_detected_colnames = []
        kp_gt_colnames = []
        for kp_idx in range(n_keypoints):
            kp_detected_colnames.append("kp{}x".format(kp_idx))
            kp_detected_colnames.append("kp{}y".format(kp_idx))
            kp_gt_colnames.append("kp{}x_gt".format(kp_idx))
            kp_gt_colnames.append("kp{}y_gt".format(kp_idx))
        header = ["name"] + kp_detected_colnames + kp_gt_colnames
        csv_writer.writerow(header)

        for name, kp_detected, kp_gt in zip(
            sample_names, keypoints_detected, keypoints_gt
        ):
            entry = (
                [name]
                + kp_detected.reshape(n_keypoint_elements).tolist()
                + kp_gt.reshape(n_keypoint_elements).tolist()
            )
            csv_writer.writerow(entry)

# write_pnp_csv: poses is expected to be array of [x y z x y z w]
def write_pnp_csv(
    pnp_path,
    sample_names,
    pnp_attempts_successful,
    poses,
    pnp_add,
    num_inframe_projs_gt,
):

    n_samples = len(sample_names)

    assert n_samples == len(pnp_attempts_successful)
    assert n_samples == len(poses)
    assert n_samples == len(num_inframe_projs_gt)
    assert n_samples == len(pnp_add)

    with open(pnp_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)

        header = [
            "name",
            "pnp_success",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_qx",
            "pose_qy",
            "pose_qz",
            "pose_qw",
            "add",
            "n_inframe_gt_projs",
        ]
        csv_writer.writerow(header)

        for name, pnp_successful, pose, this_pnp_add, this_num_inframe_projs_gt in zip(
            sample_names, pnp_attempts_successful, poses, pnp_add, num_inframe_projs_gt
        ):
            entry = (
                [name]
                + [pnp_successful]
                + pose
                + [this_pnp_add]
                + [this_num_inframe_projs_gt]
            )
            csv_writer.writerow(entry)


def keypoint_metrics(
    keypoints_detected, keypoints_gt, all_json_np, image_resolution, auc_pixel_threshold=12.0, syn=False
):

    # TBD: input argument handling
    num_gt_outframe = 0
    num_gt_inframe = 0
    num_missing_gt_outframe = 0
    num_found_gt_outframe = 0
    num_found_gt_inframe = 0
    num_missing_gt_inframe = 0
    
    N, _ = keypoints_gt.shape
    
    if syn:
        gap = 140
    else:
        gap = 0
    kp_errors = []
    for kp_proj_detect, kp_proj_gt, json_order in zip(keypoints_detected, keypoints_gt, all_json_np):

        if (
            # kp_proj_gt[0] <= 140.0
            kp_proj_gt[0] < 0.0 + gap
            # or kp_proj_gt[0] >= image_resolution[0] - 140.0
            or kp_proj_gt[0] > image_resolution[0] - gap
            or kp_proj_gt[1] < 0.0
            or kp_proj_gt[1] > image_resolution[1]
        ):
            # GT keypoint is out of frame
            num_gt_outframe += 1
            # print('gt', kp_proj_gt)
            # print('detect', kp_proj_detect)
            # print('json_order', json_order)

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (correct)
                num_missing_gt_outframe += 1
            else:
                # Found a keypoint (wrong)
                num_found_gt_outframe += 1

        else:
            # GT keypoint is in frame
            num_gt_inframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (wrong)
                num_missing_gt_inframe += 1
                # print('order', json_order)
                # print('kp_proj_gt', kp_proj_gt)
            else:
                # Found a keypoint (correct)
                num_found_gt_inframe += 1

                kp_errors.append((kp_proj_detect - kp_proj_gt).tolist())

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)

        # compute the auc
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel)
            / float(auc_pixel_threshold)
            / float(num_gt_inframe)
        )

    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None

    metrics = {
        "num_gt_outframe": num_gt_outframe,
        "num_missing_gt_outframe": num_missing_gt_outframe,
        "num_found_gt_outframe": num_found_gt_outframe,
        "num_gt_inframe": num_gt_inframe,
        "num_found_gt_inframe": num_found_gt_inframe,
        "num_missing_gt_inframe": num_missing_gt_inframe,
        "l2_error_mean_px": kp_l2_error_mean,
        "l2_error_median_px": kp_l2_error_median,
        "l2_error_std_px": kp_l2_error_std,
        "l2_error_auc": kp_auc,
        "l2_error_auc_thresh_px": auc_pixel_threshold,
    }
    return metrics


def pnp_metrics(
    pnp_add,
    num_inframe_projs_gt,
    num_min_inframe_projs_gt_for_pnp=4,
    add_auc_threshold=0.06,
    pnp_magic_number=-999.0,
):
    pnp_add = np.array(pnp_add)
    num_inframe_projs_gt = np.array(num_inframe_projs_gt)

    idx_pnp_found = np.where(pnp_add > pnp_magic_number)[0]
    add_pnp_found = pnp_add[idx_pnp_found]
    num_pnp_found = len(idx_pnp_found)

    mean_add = np.mean(add_pnp_found)
    median_add = np.median(add_pnp_found)
    std_add = np.std(add_pnp_found)
    max_add = np.max(add_pnp_found)
    min_add = np.min(add_pnp_found)

    num_pnp_possible = len(
        np.where(num_inframe_projs_gt >= num_min_inframe_projs_gt_for_pnp)[0]
    )
    num_pnp_not_found = num_pnp_possible - num_pnp_found

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)

    counts = []
    for value in add_threshold_values:
        under_threshold = len(np.where(add_pnp_found <= value)[0]) / float(
            num_pnp_possible
        )
        counts.append(under_threshold)

    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    metrics = {
        "num_pnp_found": num_pnp_found,
        "num_pnp_not_found": num_pnp_not_found,
        "num_pnp_possible": num_pnp_possible,
        "num_min_inframe_projs_gt_for_pnp": num_min_inframe_projs_gt_for_pnp,
        "pnp_magic_number": pnp_magic_number,
        "add_mean": mean_add,
        "add_median": median_add,
        "add_std": std_add,
        "add_max": max_add,
        "add_min": min_add,
        "add_auc": auc,
        "add_auc_thresh": add_auc_threshold,
    }
    return metrics


def sample_range_analysis(
    raw_images_or_image_paths,
    sample_kp_proj_detected_netout,
    sample_kp_proj_gt_netout,
    sample_belief_maps,
    sample_names,
    sample_ranks,
    image_prefix,
    output_dir,
    keypoint_names,
    images_net_input_tensor_batch,
):
    n_keypoints = len(keypoint_names)
    n_cols = int(math.ceil(n_keypoints / 2.0))

    n_sample_range = len(raw_images_or_image_paths)

    images_net_input = sgtapose.image_proc.images_from_tensor(
        images_net_input_tensor_batch
    )
    images_net_input_overlay = []

    # Assume the belief maps are in the net output frame
    net_output_res_inf = (
        sample_belief_maps[0].shape[2],
        sample_belief_maps[0].shape[1],
    )

    for (
        keypoint_projs_detected,
        keypoint_projs_gt,
        belief_maps,
        sample_name,
        sample_rank,
        image_rgb_net_input,
    ) in zip(
        sample_kp_proj_detected_netout,
        sample_kp_proj_gt_netout,
        sample_belief_maps,
        sample_names,
        sample_ranks,
        images_net_input,
    ):

        # Create belief map mosaics, with and without keypoint overlay
        # This is in the "net output" frame belief maps
        belief_maps_mosaic_path = os.path.join(
            output_dir,
            image_prefix
            + "_belief_maps_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        belief_maps_kp_mosaic_path = os.path.join(
            output_dir,
            image_prefix
            + "_belief_maps_kp_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )

        belief_map_images = sgtapose.image_proc.images_from_belief_maps(
            belief_maps, normalization_method=6
        )
        belief_maps_mosaic = sgtapose.image_proc.mosaic_images(
            belief_map_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        belief_maps_mosaic.save(belief_maps_mosaic_path)

        # This is in the "net output" frame belief maps with keypoint overlays
        belief_map_images_kp = []
        for n_kp in range(n_keypoints):
            belief_map_image_kp = sgtapose.image_proc.overlay_points_on_image(
                belief_map_images[n_kp],
                [keypoint_projs_gt[n_kp, :], keypoint_projs_detected[n_kp, :]],
                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )
            belief_map_images_kp.append(belief_map_image_kp)
        belief_maps_kp_mosaic = sgtapose.image_proc.mosaic_images(
            belief_map_images_kp, rows=2, cols=n_cols, inner_padding_px=10
        )
        belief_maps_kp_mosaic.save(belief_maps_kp_mosaic_path)

        # Create overlay of keypoints (detected and gt) on network input image
        net_input_res_inf = image_rgb_net_input.size
        scale_factor_netin_from_netout = (
            float(net_input_res_inf[0]) / float(net_output_res_inf[0]),
            float(net_input_res_inf[1]) / float(net_output_res_inf[1]),
        )

        kp_projs_detected_net_input = []
        kp_projs_gt_net_input = []
        for n_kp in range(n_keypoints):
            kp_projs_detected_net_input.append(
                [
                    keypoint_projs_detected[n_kp][0]
                    * scale_factor_netin_from_netout[0],
                    keypoint_projs_detected[n_kp][1]
                    * scale_factor_netin_from_netout[1],
                ]
            )
            kp_projs_gt_net_input.append(
                [
                    keypoint_projs_gt[n_kp][0] * scale_factor_netin_from_netout[0],
                    keypoint_projs_gt[n_kp][1] * scale_factor_netin_from_netout[1],
                ]
            )

        image_rgb_net_input_overlay = sgtapose.image_proc.overlay_points_on_image(
            image_rgb_net_input,
            kp_projs_gt_net_input,
            keypoint_names,
            annotation_color_dot="green",
            annotation_color_text="green",
        )
        image_rgb_net_input_overlay = sgtapose.image_proc.overlay_points_on_image(
            image_rgb_net_input_overlay,
            kp_projs_detected_net_input,
            keypoint_names,
            annotation_color_dot="red",
            annotation_color_text="red",
        )
        images_net_input_overlay.append(image_rgb_net_input_overlay)

        # Generate blended (net input + belief map) images
        blend_input_belief_map_images = []
        blend_input_belief_map_kp_images = []

        for n in range(len(belief_map_images)):
            # Upscale belief map to net input resolution
            belief_map_image_upscaled = belief_map_images[n].resize(
                net_input_res_inf, resample=PILImage.BILINEAR
            )

            # Increase image brightness to account for the belief map overlay
            # TBD - maybe use a mask instead
            blend_input_belief_map_image = PILImage.blend(
                belief_map_image_upscaled, image_rgb_net_input, alpha=0.5
            )
            blend_input_belief_map_images.append(blend_input_belief_map_image)

            # Overlay on the blended one directly so the annotation isn't blurred
            blend_input_belief_map_kp_image = sgtapose.image_proc.overlay_points_on_image(
                blend_input_belief_map_image,
                [kp_projs_gt_net_input[n], kp_projs_detected_net_input[n]],
                [keypoint_names[n]] * 2,
                annotation_color_dot=["green", "red"],
                annotation_color_text=["green", "red"],
                point_diameter=4,
            )
            blend_input_belief_map_kp_images.append(blend_input_belief_map_kp_image)

        mosaic_blend_input_belief_map_images = sgtapose.image_proc.mosaic_images(
            blend_input_belief_map_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        mosaic_blend_input_belief_map_images_path = os.path.join(
            output_dir,
            image_prefix + "_blend_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        mosaic_blend_input_belief_map_images.save(
            mosaic_blend_input_belief_map_images_path
        )

        mosaic_blend_input_belief_map_kp_images = sgtapose.image_proc.mosaic_images(
            blend_input_belief_map_kp_images, rows=2, cols=n_cols, inner_padding_px=10
        )
        mosaic_blend_input_belief_map_kp_images_path = os.path.join(
            output_dir,
            image_prefix
            + "_blend_kp_rank_{}_id_{}.png".format(sample_rank, sample_name),
        )
        mosaic_blend_input_belief_map_kp_images.save(
            mosaic_blend_input_belief_map_kp_images_path
        )

    # This just a mosaic of all the inputs in raw form
    mosaic = sgtapose.image_proc.mosaic_images(
        raw_images_or_image_paths, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_path = os.path.join(output_dir, image_prefix + ".png")
    mosaic.save(mosaic_path)

    # This is a mosaic of the net input images, with and without KP overlays
    mosaic_net_input = sgtapose.image_proc.mosaic_images(
        images_net_input, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_net_input_path = os.path.join(output_dir, image_prefix + "_net_input.png")
    mosaic_net_input.save(mosaic_net_input_path)

    mosaic_net_input_overlay = sgtapose.image_proc.mosaic_images(
        images_net_input_overlay, rows=1, cols=n_sample_range, inner_padding_px=10
    )
    mosaic_net_input_overlay_path = os.path.join(
        output_dir, image_prefix + "_net_input_kp.png"
    )
    mosaic_net_input_overlay.save(mosaic_net_input_overlay_path)
