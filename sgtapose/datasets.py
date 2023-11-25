# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
from lib.utils.image import gaussian_radius, draw_umich_gaussian
import copy
import cv2

import sgtapose

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
class CenterTrackSeqDataset(TorchDataset):
    def __init__(
        self,
        ndds_seq_dataset,
        manipulator_name, 
        keypoint_names,
        opt, 
        mean, 
        std, 
        include_ground_truth=True,
        include_belief_maps=False,
        seq_frame = False,
    ): 
        self.ndds_seq_dataset_data = ndds_seq_dataset
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.opt = opt
        self.input_w, self.input_h = self.opt.input_w, self.opt.input_h
        self.output_w, self.output_h = self.opt.output_w, self.opt.output_h
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.seq_frame = seq_frame
        self.seq_count_all = self.__len__()
        self.black_count = 0
        self.camera_K = np.array([[502.30, 0.0, 319.75], [0.0, 502.30, 179.75], [0.0, 0.0, 1.0]])
        print('seq_count_all', self.seq_count_all)
        

        
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        
    
    def __len__(self):
        return len(self.ndds_seq_dataset_data)
    
    def __getitem__(self, index):
        datum = self.ndds_seq_dataset_data[index]
        if self.seq_frame:
            Frame, ind = datum["next_frame_name"].split('/')
            ind = int(ind)
            if ind % self.seq_frame == 0:
                # 
                next_frame_name = datum["prev_frame_name"]
                next_frame_img_path = datum["prev_frame_img_path"]
                next_frame_data_path = datum["prev_frame_data_path"]
                prev_frame_name = '/'.join([Frame, str(ind - self.seq_frame).zfill(4)])
                old_name = str(ind).zfill(4)
                new_name = str(ind - self.seq_frame).zfill(4)

                
                prev_frame_img_path = datum["next_frame_img_path"].replace(old_name + "_color.png", new_name+"_color.png")
                prev_frame_data_path = datum["next_frame_data_path"].replace(old_name + "_meta.json", new_name + "_meta.json")
                if self.opt.phase == "CenterNet":
                    next_frame_name = prev_frame_name
                    next_frame_img_path = prev_frame_img_path
                    next_frame_data_path = prev_frame_data_path
                    
                    assert next_frame_img_path == prev_frame_img_path
                    assert next_frame_data_path == prev_frame_data_path
            else:
                prev_frame_name = datum["prev_frame_name"]
                prev_frame_img_path = datum["prev_frame_img_path"]
                prev_frame_data_path = datum["prev_frame_data_path"]
                next_frame_name = datum["next_frame_name"]
                next_frame_img_path = datum["next_frame_img_path"]
                next_frame_data_path = datum["next_frame_data_path"]
        
        if self.include_ground_truth:
            prev_keypoints = sgtapose.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names, self.camera_K)
            next_keypoints = sgtapose.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, self.keypoint_names, self.camera_K)
        else:
            prev_keypoints = sgtapose.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, [], self.camera_K)
            next_keypoints = sgtapose.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, [], self.camera_K)
        
        # load iamge and transform to network input resolution --pre augmentation
        prev_image_rgb_raw = cv2.imread(prev_frame_img_path)
        next_image_rgb_raw = cv2.imread(next_frame_img_path)
        assert prev_image_rgb_raw.shape == next_image_rgb_raw.shape
        
        height, width, _ = prev_image_rgb_raw.shape
        c = np.array([prev_image_rgb_raw.shape[1] / 2., prev_image_rgb_raw.shape[0] / 2.], dtype=np.float32) # (width/2, height/2)
        s = max(prev_image_rgb_raw.shape[0], prev_image_rgb_raw.shape[1]) * 1.0
        aug_s, rot = 1.0, 0 
        c, aug_s = sgtapose.utilities._get_aug_param(c, s, width, height)
        s = s * aug_s
        
        trans_input = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.input_w, self.input_h]) 
        trans_output = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.output_w, self.output_h])
        prev_image_rgb_net_input = sgtapose.utilities._get_input(prev_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std) # 3 x H x W
        next_image_rgb_net_input = sgtapose.utilities._get_input(next_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std)
        assert prev_image_rgb_net_input.shape == next_image_rgb_net_input.shape
        assert prev_image_rgb_net_input.shape == (3, self.input_h, self.input_w)
        prev_image_rgb_net_input_as_tensor = torch.from_numpy(prev_image_rgb_net_input).float()
        next_image_rgb_net_input_as_tensor = torch.from_numpy(next_image_rgb_net_input).float()
        
        prev_kp_projs_raw_np = np.array(prev_keypoints["projections"], dtype=np.float32)
        next_kp_projs_raw_np = np.array(next_keypoints["projections"], dtype=np.float32)
        prev_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(prev_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        next_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(next_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        
        # Convert keypoint data to tensors -use float32 size 
        prev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_cam"])
                ).float()
        
        prev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_robot"])
                ).float() 
                
        prev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(prev_kp_projs_net_output_np)
                ).float()
        # prev_kp_projs_net_output_as_tensor_int = prev_kp_projs_net_output_as_tensor.int()
        
        
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output_np) 
                ).float()
        next_kp_projs_net_output_as_tensor_int = sgtapose.utilities.make_int(next_kp_projs_net_output_as_tensor, [self.output_w, self.output_h])
        next_kp_projs_net_output_int_np = next_kp_projs_net_output_as_tensor_int.numpy()
        
        sample = {
            "prev_image_raw_path" : prev_frame_img_path, 
            "prev_image_rgb_input" : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,
            "next_image_raw_path" : next_frame_img_path, 
            "next_image_rgb_input" : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,
            "next_keypoint_projections_output_int": next_kp_projs_net_output_as_tensor_int, 
            "reg" : next_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "tracking" : prev_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "config" : datum}
        
        
        if self.include_belief_maps:
            prev_origin_maps_as_whole_np = sgtapose.utilities.get_prev_hm(prev_kp_projs_raw_np, trans_input,self.input_w, self.input_h, width, height, hm_disturb = self.opt.hm_disturb, lost_disturb=self.opt.lost_disturb) 
            prev_origin_maps_as_whole_as_tensor = torch.from_numpy(prev_origin_maps_as_whole_np).float()
            sample["prev_origin_belief_maps"] = prev_origin_maps_as_whole_as_tensor
            
            next_belief_maps = sgtapose.utilities.get_hm(next_kp_projs_net_output_int_np, self.output_w, self.output_h)
            next_belief_maps_as_tensor = torch.from_numpy(next_belief_maps).float()
            sample["next_belief_maps"] = next_belief_maps_as_tensor
            
            prev_kp_pos_gt_np = prev_keypoint_positions_wrt_robot_as_tensor.numpy()
            next_kp_pos_gt_np = next_keypoint_positions_wrt_robot_as_tensor.numpy()
            prev_kp_projs_gt = np.array(prev_keypoints["projections"], dtype=np.float64)
            pnp_retval, next_kp_projs_est, prev_kp_projs_noised_np = sgtapose.geometric_vision.get_pnp_keypoints(prev_kp_pos_gt_np, prev_kp_projs_gt, next_kp_pos_gt_np, self.camera_K, self.opt.hm_disturb, self.opt.lost_disturb) 

            
            prev_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_noised_np, trans_input, self.input_w, self.input_h, \
            width, height)
            prev_belief_maps_as_whole_as_tensor = torch.from_numpy(prev_belief_maps_as_whole_np).float()
            sample["prev_belief_maps"] = prev_belief_maps_as_whole_as_tensor
            
            repro_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,self.input_w, self.input_h, \
            width, height)  
            repro_belief_maps_as_whole_as_tensor = torch.from_numpy(repro_belief_maps_as_whole_np).float()
            sample["repro_belief_maps"] = repro_belief_maps_as_whole_as_tensor
            
            prev_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_noised_np, prev_kp_pos_gt_np, trans_output, self.output_w,self.output_h,width, height)
            prev_belief_maps_cls_as_tensor = torch.from_numpy(prev_belief_maps_cls_np).float()
            sample["prev_belief_maps_cls"] = prev_belief_maps_cls_as_tensor
            
            repro_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_kp_pos_gt_np, trans_output, self.output_w, self.output_h, width, height)
            repro_belief_maps_cls_as_tensor = torch.from_numpy(repro_belief_maps_cls_np).float()
            sample["repro_belief_maps_cls"] = repro_belief_maps_cls_as_tensor
 
        return sample

class CenterTrackThreeDataset(TorchDataset):
    def __init__(
        self,
        ndds_three_dataset,
        manipulator_name, 
        keypoint_names,
        opt, 
        mean, 
        std, 
        include_ground_truth=True,
        include_belief_maps=False,
        seq_frame = False,
    ): 
        self.ndds_three_dataset_data = ndds_three_dataset
        # self.ndds_seq_dataset_config = dataset_config
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.opt = opt
        self.input_w, self.input_h = self.opt.input_w, self.opt.input_h
        self.output_w, self.output_h = self.opt.output_w, self.opt.output_h
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.seq_frame = seq_frame
        self.seq_count_all = self.__len__()
        self.black_count = 0
        self.camera_K = np.array([[502.30, 0.0, 319.5], [0.0, 502.30, 179.5], [0.0, 0.0, 1.0]])
        print('seq_count_all', self.seq_count_all)
        
        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        
    
    def __len__(self):
        return len(self.ndds_three_dataset_data)
    
    def __getitem__(self, index):
        # {prev_frame_name; prev_frame_img_path, 
        # prev_frame_data_path, next_frame_name, next_frame_img_path, next_frame_data_path}
        
        # Parse this datum
        datum = self.ndds_three_dataset_data[index]
        # print('datum', datum)
        
        pprev_frame_name = datum["pprev_frame_name"]
        pprev_frame_img_path = datum["pprev_frame_img_path"]
        pprev_frame_data_path = datum["pprev_frame_data_path"]
        prev_frame_name = datum["prev_frame_name"]
        prev_frame_img_path = datum["prev_frame_img_path"]
        prev_frame_data_path = datum["prev_frame_data_path"]
        next_frame_name = datum["next_frame_name"]
        next_frame_img_path = datum["next_frame_img_path"]
        next_frame_data_path = datum["next_frame_data_path"]
        
        
        if self.include_ground_truth:
            pprev_keypoints = sgtapose.utilities.load_seq_keypoints(pprev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
            prev_keypoints = sgtapose.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
            next_keypoints = sgtapose.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
        else:
            pprev_keypoints = sgtapose.utilities.load_seq_keypoints(pprev_frame_data_path, \
                            self.manipulator_name, [])
            prev_keypoints = sgtapose.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, [])
            next_keypoints = sgtapose.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, [])
        
        pprev_image_rgb_raw = cv2.imread(pprev_frame_img_path)
        prev_image_rgb_raw = cv2.imread(prev_frame_img_path)
        next_image_rgb_raw = cv2.imread(next_frame_img_path)
        assert prev_image_rgb_raw.shape == next_image_rgb_raw.shape
        assert pprev_image_rgb_raw.shape == next_image_rgb_raw.shape
        
        height, width, _ = prev_image_rgb_raw.shape
        c = np.array([prev_image_rgb_raw.shape[1] / 2., prev_image_rgb_raw.shape[0] / 2.], dtype=np.float32) # (width/2, height/2)
        s = max(prev_image_rgb_raw.shape[0], prev_image_rgb_raw.shape[1]) * 1.0
        aug_s, rot = 1.0, 0 
        c, aug_s = sgtapose.utilities._get_aug_param(c, s, width, height)
        s = s * aug_s

        trans_input = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.input_w, self.input_h]) 
        trans_output = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.output_w, self.output_h])
        
        pprev_image_rgb_net_input = sgtapose.utilities._get_input(pprev_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std) 
        prev_image_rgb_net_input = sgtapose.utilities._get_input(prev_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std) # 3 x H x W
        next_image_rgb_net_input = sgtapose.utilities._get_input(next_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std)
        assert pprev_image_rgb_net_input.shape == next_image_rgb_net_input.shape
        assert prev_image_rgb_net_input.shape == next_image_rgb_net_input.shape
        assert prev_image_rgb_net_input.shape == (3, self.input_h, self.input_w)
        pprev_image_rgb_net_input_as_tensor = torch.from_numpy(pprev_image_rgb_net_input).float()
        prev_image_rgb_net_input_as_tensor = torch.from_numpy(prev_image_rgb_net_input).float()
        next_image_rgb_net_input_as_tensor = torch.from_numpy(next_image_rgb_net_input).float()
        
        pprev_kp_projs_raw_np = np.array(pprev_keypoints["projections"], dtype=np.float32)
        prev_kp_projs_raw_np = np.array(prev_keypoints["projections"], dtype=np.float32)
        next_kp_projs_raw_np = np.array(next_keypoints["projections"], dtype=np.float32)
        pprev_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(pprev_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        prev_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(prev_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        next_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(next_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        
        # Convert keypoint data to tensors -use float32 size 
        prev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_cam"])
                ).float()
        
        prev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_robot"])
                ).float() 
                
        prev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(prev_kp_projs_net_output_np)
                ).float()

        pprev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(pprev_keypoints["positions_wrt_cam"])
                ).float()
        
        pprev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(pprev_keypoints["positions_wrt_robot"])
                ).float() 
                
        pprev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(pprev_kp_projs_net_output_np)
                ).float()
        
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output_np) 
                ).float()
                
        next_kp_projs_net_output_as_tensor_int = sgtapose.utilities.make_int(next_kp_projs_net_output_as_tensor, [self.output_w, self.output_h])
        next_kp_projs_net_output_int_np = next_kp_projs_net_output_as_tensor_int.numpy()
        
        sample = {}
        sample.update({"pprev_image_raw_path" : pprev_frame_img_path, 
            "pprev_image_rgb_input" : pprev_image_rgb_net_input_as_tensor,
            "pprev_keypoint_projections_output": pprev_kp_projs_net_output_as_tensor,
            "pprev_keypoint_positions_wrt_cam": pprev_keypoint_positions_wrt_cam_as_tensor,
            "pprev_keypoint_positions_wrt_robot" : pprev_keypoint_positions_wrt_robot_as_tensor,})
       
        sample.update({"prev_image_raw_path" : prev_frame_img_path, 
            "prev_image_rgb_input" : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,})
        
        sample.update({"next_image_raw_path" : next_frame_img_path, 
            "next_image_rgb_input" : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,})

        sample.update({"next_keypoint_projections_output_int": next_kp_projs_net_output_as_tensor_int, 
            "reg" : next_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "tracking" : prev_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "config" : datum})
        
        
        if self.include_belief_maps:
            prev_origin_maps_as_whole_np = sgtapose.utilities.get_prev_hm(prev_kp_projs_raw_np, trans_input,self.input_w, self.input_h, width, height, hm_disturb = self.opt.hm_disturb, lost_disturb=self.opt.lost_disturb) 
            prev_origin_maps_as_whole_as_tensor = torch.from_numpy(prev_origin_maps_as_whole_np).float()
            sample["prev_origin_belief_maps"] = prev_origin_maps_as_whole_as_tensor
            
            next_belief_maps = sgtapose.utilities.get_hm(next_kp_projs_net_output_int_np, self.output_w, self.output_h)
            next_belief_maps_as_tensor = torch.from_numpy(next_belief_maps).float()
            sample["next_belief_maps"] = next_belief_maps_as_tensor
            
            camera_K = np.array([[502.30, 0.0, 319.5], [0.0, 502.30, 179.5], [0.0, 0.0, 1.0]])
            pprev_kp_pos_gt_np = pprev_keypoint_positions_wrt_robot_as_tensor.numpy()
            prev_kp_pos_gt_np = prev_keypoint_positions_wrt_robot_as_tensor.numpy()
            next_kp_pos_gt_np = next_keypoint_positions_wrt_robot_as_tensor.numpy()
            pprev_kp_projs_gt = np.array(pprev_keypoints["projections"], dtype=np.float64)
            prev_kp_projs_gt = np.array(prev_keypoints["projections"], dtype=np.float64)            
            pnp_retval, pprev_kp_projs_noised_np, prev_kp_projs_est, next_kp_projs_est = sgtapose.geometric_vision.get_three_pnp_keypoints(
            pprev_kp_pos_gt_np, pprev_kp_projs_gt, prev_kp_pos_gt_np, prev_kp_projs_gt, next_kp_pos_gt_np, camera_K, self.opt.hm_disturb, self.opt.lost_disturb)
            
            
            pprev_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(pprev_kp_projs_noised_np, trans_input, self.input_w, self.input_h, \
            width, height)
            pprev_belief_maps_as_whole_as_tensor = torch.from_numpy(pprev_belief_maps_as_whole_np).float()
            sample["pprev_belief_maps"] = pprev_belief_maps_as_whole_as_tensor
            
            prev_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_est, trans_input, self.input_w, self.input_h, \
            width, height)
            prev_belief_maps_as_whole_as_tensor = torch.from_numpy(prev_belief_maps_as_whole_np).float()
            sample["prev_belief_maps"] = prev_belief_maps_as_whole_as_tensor
            
            repro_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,self.input_w, self.input_h, \
            width, height)  
            repro_belief_maps_as_whole_as_tensor = torch.from_numpy(repro_belief_maps_as_whole_np).float()
            sample["repro_belief_maps"] = repro_belief_maps_as_whole_as_tensor
            
            pprev_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(pprev_kp_projs_noised_np, pprev_kp_pos_gt_np, trans_output, self.output_w,self.output_h,width, height)
            pprev_belief_maps_cls_as_tensor = torch.from_numpy(pprev_belief_maps_cls_np).float()
            sample["pprev_belief_maps_cls"] = pprev_belief_maps_cls_as_tensor
            
            prev_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_est, prev_kp_pos_gt_np, trans_output, self.output_w,self.output_h,width, height)
            prev_belief_maps_cls_as_tensor = torch.from_numpy(prev_belief_maps_cls_np).float()
            sample["prev_belief_maps_cls"] = prev_belief_maps_cls_as_tensor
            
            repro_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_kp_pos_gt_np, trans_output, self.output_w, self.output_h, width, height)
            repro_belief_maps_cls_as_tensor = torch.from_numpy(repro_belief_maps_cls_np).float()
            sample["repro_belief_maps_cls"] = repro_belief_maps_cls_as_tensor
 
        return sample




class CenterTrackSeqDepthDataset(TorchDataset):
    def __init__(
        self,
        ndds_seq_dataset,
        manipulator_name, 
        keypoint_names,
        opt, 
        mean, 
        std, 
        include_ground_truth=True,
        include_belief_maps=False,
        seq_frame = False,
    ): 
        self.ndds_seq_dataset_data = ndds_seq_dataset
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.opt = opt
        self.input_w, self.input_h = self.opt.input_w, self.opt.input_h
        self.output_w, self.output_h = self.opt.output_w, self.opt.output_h
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.seq_frame = seq_frame
        self.seq_count_all = self.__len__()
        self.black_count = 0
        self.camera_K = np.array([[502.30, 0.0, 319.75], [0.0, 502.30, 179.75], [0.0, 0.0, 1.0]])
        print('seq_count_all', self.seq_count_all)
        
        
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        
    
    def __len__(self):
        return len(self.ndds_seq_dataset_data)
    
    def __getitem__(self, index):
        datum = self.ndds_seq_dataset_data[index]
        if self.seq_frame:
            Frame, ind = datum["next_frame_name"].split('/')
            ind = int(ind)
            if ind % self.seq_frame == 0:
                next_frame_name = datum["prev_frame_name"]
                next_frame_img_path = datum["prev_frame_img_path"]
                next_frame_data_path = datum["prev_frame_data_path"]
                prev_frame_name = '/'.join([Frame, str(ind - self.seq_frame).zfill(4)])
                old_name = str(ind).zfill(4)
                new_name = str(ind - self.seq_frame).zfill(4)

                
                prev_frame_img_path = datum["next_frame_img_path"].replace(old_name + "_color.png", new_name+"_color.png")
                prev_frame_data_path = datum["next_frame_data_path"].replace(old_name + "_meta.json", new_name + "_meta.json")
                if self.opt.phase == "CenterNet":
                    next_frame_name = prev_frame_name
                    next_frame_img_path = prev_frame_img_path
                    next_frame_data_path = prev_frame_data_path
                    
                    assert next_frame_img_path == prev_frame_img_path
                    assert next_frame_data_path == prev_frame_data_path
            else:
                prev_frame_name = datum["prev_frame_name"]
                prev_frame_img_path = datum["prev_frame_img_path"]
                prev_frame_data_path = datum["prev_frame_data_path"]
                next_frame_name = datum["next_frame_name"]
                next_frame_img_path = datum["next_frame_img_path"]
                next_frame_data_path = datum["next_frame_data_path"]
        
        if self.include_ground_truth:
            prev_keypoints = sgtapose.utilities.load_depth_keypoints(prev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names, self.camera_K)
            next_keypoints = sgtapose.utilities.load_depth_keypoints(next_frame_data_path, \
                            self.manipulator_name, self.keypoint_names, self.camera_K)
        else:
            prev_keypoints = sgtapose.utilities.load_depth_keypoints(prev_frame_data_path, \
                            self.manipulator_name, [], self.camera_K)
            next_keypoints = sgtapose.utilities.load_depth_keypoints(next_frame_data_path, \
                            self.manipulator_name, [], self.camera_K)
        

        prev_image_rgb_raw = cv2.imread(prev_frame_img_path)
        next_image_rgb_raw = cv2.imread(next_frame_img_path)
        assert prev_image_rgb_raw.shape == next_image_rgb_raw.shape
        
        height, width, _ = prev_image_rgb_raw.shape
        c = np.array([prev_image_rgb_raw.shape[1] / 2., prev_image_rgb_raw.shape[0] / 2.], dtype=np.float32) # (width/2, height/2)
        s = max(prev_image_rgb_raw.shape[0], prev_image_rgb_raw.shape[1]) * 1.0
        aug_s, rot = 1.0, 0 
        c, aug_s = sgtapose.utilities._get_aug_param(c, s, width, height)
        s = s * aug_s
        
        trans_input = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.input_w, self.input_h]) 
        trans_output = sgtapose.utilities.get_affine_transform(
        c, s, rot, [self.output_w, self.output_h])
        prev_image_rgb_net_input = sgtapose.utilities._get_input(prev_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std) # 3 x H x W
        next_image_rgb_net_input = sgtapose.utilities._get_input(next_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std)
        assert prev_image_rgb_net_input.shape == next_image_rgb_net_input.shape
        assert prev_image_rgb_net_input.shape == (3, self.input_h, self.input_w)
        prev_image_rgb_net_input_as_tensor = torch.from_numpy(prev_image_rgb_net_input).float()
        next_image_rgb_net_input_as_tensor = torch.from_numpy(next_image_rgb_net_input).float()
        
        prev_kp_projs_raw_np = np.array(prev_keypoints["projections"], dtype=np.float32)
        next_kp_projs_raw_np = np.array(next_keypoints["projections"], dtype=np.float32)
        prev_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(prev_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        next_kp_projs_net_output_np = sgtapose.utilities.affine_transform_and_clip(next_kp_projs_raw_np, trans_output, self.output_w, self.output_h,\
        width, height)
        
        # Convert keypoint data to tensors -use float32 size 
        prev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_cam"])
                ).float()
        
        prev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_robot"])
                ).float() 
                
        prev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(prev_kp_projs_net_output_np)
                ).float()
        
        
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output_np) 
                ).float()
        next_kp_projs_net_output_as_tensor_int = sgtapose.utilities.make_int(next_kp_projs_net_output_as_tensor, [self.output_w, self.output_h])
        next_kp_projs_net_output_int_np = next_kp_projs_net_output_as_tensor_int.numpy()
        
        sample = {
            "prev_image_raw_path" : prev_frame_img_path, 
            "prev_image_rgb_input" : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,
            "next_image_raw_path" : next_frame_img_path, 
            "next_image_rgb_input" : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,
            "next_keypoint_projections_output_int": next_kp_projs_net_output_as_tensor_int, 
            "reg" : next_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "tracking" : prev_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "config" : datum}
        
        

        if self.include_belief_maps:
            prev_origin_maps_as_whole_np = sgtapose.utilities.get_prev_hm(prev_kp_projs_raw_np, trans_input,self.input_w, self.input_h, width, height, hm_disturb = self.opt.hm_disturb, lost_disturb=self.opt.lost_disturb) 
            prev_origin_maps_as_whole_as_tensor = torch.from_numpy(prev_origin_maps_as_whole_np).float()
            sample["prev_origin_belief_maps"] = prev_origin_maps_as_whole_as_tensor
            
            next_belief_maps = sgtapose.utilities.get_hm(next_kp_projs_net_output_int_np, self.output_w, self.output_h)
            next_belief_maps_as_tensor = torch.from_numpy(next_belief_maps).float()
            sample["next_belief_maps"] = next_belief_maps_as_tensor
            

            prev_kp_pos_gt_np = prev_keypoint_positions_wrt_robot_as_tensor.numpy()
            next_kp_pos_gt_np = next_keypoint_positions_wrt_robot_as_tensor.numpy()
            prev_kp_projs_gt = np.array(prev_keypoints["projections"], dtype=np.float64)
            pnp_retval, next_kp_projs_est, prev_kp_projs_noised_np = sgtapose.geometric_vision.get_pnp_keypoints(prev_kp_pos_gt_np, prev_kp_projs_gt, next_kp_pos_gt_np, self.camera_K, self.opt.hm_disturb, self.opt.lost_disturb) 
            
            prev_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_noised_np, trans_input, self.input_w, self.input_h, \
            width, height)
            prev_belief_maps_as_whole_as_tensor = torch.from_numpy(prev_belief_maps_as_whole_np).float()
            sample["prev_belief_maps"] = prev_belief_maps_as_whole_as_tensor
            
            repro_belief_maps_as_whole_np = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,self.input_w, self.input_h, \
            width, height)  
            repro_belief_maps_as_whole_as_tensor = torch.from_numpy(repro_belief_maps_as_whole_np).float()
            sample["repro_belief_maps"] = repro_belief_maps_as_whole_as_tensor
            
            prev_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_noised_np, prev_kp_pos_gt_np, trans_output, self.output_w,self.output_h,width, height)
            prev_belief_maps_cls_as_tensor = torch.from_numpy(prev_belief_maps_cls_np).float()
            sample["prev_belief_maps_cls"] = prev_belief_maps_cls_as_tensor
            
            repro_belief_maps_cls_np = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_kp_pos_gt_np, trans_output, self.output_w, self.output_h, width, height)
            repro_belief_maps_cls_as_tensor = torch.from_numpy(repro_belief_maps_cls_np).float()
            sample["repro_belief_maps_cls"] = repro_belief_maps_cls_as_tensor
 
        return sample


