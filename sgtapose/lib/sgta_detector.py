# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 00:33:32 2022

@author: lenovo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math 
import os
from ruamel.yaml import YAML
from copy import deepcopy
from PIL import Image, ImageEnhance

from .model.model import create_model, load_model, create_dream_hourglass
from .model.decode import dream_generic_decode
from .model.utils import flip_tensor, flip_lr_off, flip_lr
from .utils.image import get_affine_transform, affine_transform, transform_preds_with_trans
from .utils.image import dream_draw_umich_gaussian, gaussian_radius
from .utils.post_process import dream_generic_post_process
from .utils.debugger import Debugger
from .utils.tracker import Tracker
import sgtapose
#import numba
#from numba import jit

class SGTADetector(object):
    def __init__(self, opt,keypoint_names, is_real, is_ct, idx=None):
        if opt.gpus[0] >= 0:
            opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")
        
        print("Creating model ...")
        
        if is_ct:
            self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
            self.model = load_model(self.model, opt.load_model, opt)
            self.model = self.model.to(opt.device)
        else:
            self.model = torch.nn.DataParallel(create_dream_hourglass(opt.num_classes, opt.dream_mode, opt.dream_deconv_decoder))
            self.model.load_state_dict(torch.load(opt.load_model))   
            self.model = self.model.to(opt.device)
        
        self.model.eval()
        

        self.opt = opt
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 1, 3)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 1, 3)
        self.pause = not opt.no_pause
        self.idx = idx
        self.rest_focal_length =  self.opt.test_focal_length
        # self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.tracker = Tracker(opt)
        self.is_real = is_real
        self.is_ct = is_ct
        self.phase = opt.phase
        self.dataset_path = self.opt.infer_dataset
        self.output_dir =  opt.save_dir
        self.pre_hm_teaser = opt.save_dir
        sgtapose.utilities.exists_or_mkdir(self.output_dir)
        
        self.keypoint_names = keypoint_names
        if self.is_real and not self.opt.is_real_ros:
            self.camera_data_path = os.path.join(self.dataset_path, self.is_real, "_camera_settings.json")
            print('camera_data_path', self.camera_data_path)
            self.camera_K = self.load_camera_intrinsics(self.camera_data_path)
            print('camera_K', self.camera_K)
        else:
            self.camera_K = np.array([[502.30, 0.0, 319.75], [0.0, 502.30, 179.75], [0.0, 0.0, 1.0]])
        
        if self.opt.is_real_ros:
            print("multi_frame_real_ros", self.opt.is_real_ros)
            self.camera_data_path = os.path.join(self.dataset_path, self.is_real, "_camera_settings.json")
            self.camera_K = self.load_camera_intrinsics(self.camera_data_path)
        print("self.camera_K", self.camera_K)

    
    def load_camera_intrinsics(self, camera_data_path):
        
        # Input argument handling
        assert os.path.exists(
            camera_data_path
        ), 'Expected path "{}" to exist, but it does not.'.format(camera_data_path)
    
        # Create YAML/json parser
        data_parser = YAML(typ="safe")
    
        with open(camera_data_path, "r") as f:
            cam_settings_data = data_parser.load(f.read().replace('\t',''))
        
    
        camera_fx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fx"]
        camera_fy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["fy"]
        camera_cx = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cx"]
        camera_cy = cam_settings_data["camera_settings"][0]["intrinsic_settings"]["cy"]
        camera_K = np.array(
            [[camera_fx, 0.0, camera_cx], [0.0, camera_fy, camera_cy], [0.0, 0.0, 1.0]]
        )
    
        return camera_K
        
    
    def run(self, image_or_path_or_tensor, i, json_path, meta={}, is_final=False, save_dir=None, teaser_flag=False, img_path=None):
        self.img_path = img_path
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
        # self.debugger.clear()
        start_time = time.time()
        
        # read image
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor 
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True
        
        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        output_detections = []
        
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta, i)
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
                if 'pre_dets' in pre_processed_images['meta']:
                    meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
                if 'cur_dets' in pre_processed_images['meta']:
                    meta['cur_dets'] = pre_processed_images['meta']['cur_dets']
            
            images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
            pre_hms, pre_inds = None, None
            if self.opt.tracking:
                if self.pre_images is None:
                    print('Initialize tracking!')
                    self.pre_images = images 
                    self.pre_json_path = json_path
                    self.tracker.init_track(
                        meta['pre_dets'] if 'pre_dets' in meta else [])
                if self.opt.pre_hm:
                    if i == 0:
                        pre_hms, repro_hms, pre_hms_cls, repro_hms_cls, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm) 
                    else:
                        _, _, _, _, pre_inds = self._get_additional_inputs(
                         self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
                        if self.is_real:
                            pre_hms, repro_hms, pre_hms_cls, repro_hms_cls = self._get_further_dt_pnp_inputs_real(self.detected_kps, meta, self.pre_json_path, json_path)
                            if self.phase == "CenterTrack":
                                pre_hms = self._get_further_dt_inputs(self.detected_kps, meta)
                        else:
                            pre_hms, repro_hms, pre_hms_cls, repro_hms_cls = self._get_further_dt_pnp_inputs(self.detected_kps, meta, self.pre_json_path, json_path)
                            if self.phase == "CenterTrack":
                                pre_hms = self._get_further_dt_inputs(self.detected_kps, meta)

                                   
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time         
            output, dets, forward_time, hms = self.process(
              images, self.pre_images, pre_hms, repro_hms, pre_inds, return_time=True, pre_hms_cls=pre_hms_cls, repro_hms_cls=repro_hms_cls)
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
            result = self.post_process(dets, meta, scale)
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(result)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        
        if self.opt.tracking and self.opt.is_ct:
            # public detection mode in MOT challenge
            public_det = meta['cur_dets'] if self.opt.public_det else None
            # add tracking id to results
            results = self.tracker.step(results, public_det)
        self.pre_images = images
        self.pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print("image.shape", self.pre_image.shape)
        self.pre_image = Image.fromarray(np.uint8(self.pre_image)) 



        tracking_time = time.time()
        track_time += tracking_time - end_time
        tot_time += tracking_time - start_time
        self.cnt += 1

        show_results_time = time.time()
        display_time += show_results_time - end_time
        
        # return results and run time
        ret = {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'track': track_time,
                'display': display_time}
        if self.opt.save_video:
            try:
                # return debug image for saving video
                ret.update({'generic': self.debugger.imgs['generic']}) 
            except:
                pass
        
        if teaser_flag and i >= 2:
          self._get_teaser_imgs(self.pre_hm_teaser, self.pre_image, self.detected_kps, meta, self.output_dir, i) 
        self.pre_json_path = json_path
        self.pre_img_path = img_path
        self.detected_kps = self._get_final_kps(results)       

        return ret, self.detected_kps, self.camera_K

    
    def _get_teaser_imgs(self, hms, img, detected_kps, meta, output_dir, i):        
        if self.is_ct:
            detected_kps = detected_kps.tolist()
            gt_kps_raw = []
            hms_imgs = []
            trans = get_affine_transform(meta["c"], meta["s"], 0, (meta["inp_width"], meta["inp_height"]), inv=1)
            origin_hms = cv2.warpAffine(hms[0][0].cpu().numpy(), trans, (meta["width"], meta["height"]),flags=cv2.INTER_LINEAR)
            origin_hms_tensor = torch.from_numpy(origin_hms).to(self.opt.device)
            origin_hms_img = sgtapose.image_proc.image_from_belief_map(origin_hms_tensor, normalization_method=6)
            origin_hms_np = np.asarray(origin_hms_img)
            blend_img = Image.blend(origin_hms_img, img, alpha=0.5)
        else:
            img = Image.open(self.pre_img_path).convert("RGB")
            dt_kps_raw = detected_kps
            origin_hms = sgtapose.utilities.get_prev_hm_wo_noise_dream(dt_kps_raw, img.size[0], img.size[1])
            origin_hms_tensor = torch.from_numpy(origin_hms).to(self.opt.device)
            origin_hms_img = sgtapose.image_proc.image_from_belief_map(origin_hms_tensor, normalization_method=6)
            blend_img = Image.blend(origin_hms_img, img, alpha=0.5)
        
        x3d_list = sgtapose.utilities.load_x3d(self.pre_json_path, self.keypoint_names)
        x3d_wrt_cam_np = np.array(x3d_list)
        x3d_proj_np = (self.camera_K @ x3d_wrt_cam_np.T).T
        n_kp, _ = x3d_proj_np.shape
        gt_kp_raws = np.zeros((n_kp, 2))
        gt_kp_raws[:, 0] = x3d_proj_np[:, 0] / x3d_proj_np[:, 2]
        gt_kp_raws[:, 1] = x3d_proj_np[:, 1] / x3d_proj_np[:, 2]


        for idx, gt_kp_raw in enumerate(gt_kp_raws):
            ct_gt = [0,0]
            ct_gt[0] = np.clip(gt_kp_raw[0], 0, 640-1)
            ct_gt[1] = np.clip(gt_kp_raw[1], 0, 360-1)
            blend_img = sgtapose.image_proc.overlay_points_on_image(
                        blend_img,
                        [ct_gt],
                        annotation_color_dot=["red"],
                        point_diameter=6,
                        )
                        
        save_dir = os.path.join(output_dir, str(self.idx).zfill(2))
        self.exists_or_mkdir(save_dir)
        img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_img.png"))
        blend_img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_blend_img.png"))
        origin_hms_img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_hms.png"))
    
    def blend_imgs_and_hms(self, img, hms, detected_kps, meta, output_dir, i):
        trans = get_affine_transform(meta["c"], meta["s"], 0, (meta["inp_width"], meta["inp_height"]), inv=1)
        hms = cv2.warpAffine(hms[0][0].cpu().numpy(), trans, (meta["width"], meta["height"]),flags=cv2.INTER_LINEAR)
        thresh = 0.3
        index_full = np.where(hms > thresh)
        index_zero = np.where(hms <= thresh)
        hms[index_zero] = 0
        origin_hms_tensor = torch.from_numpy(hms).to(self.opt.device)
        origin_hms_img = sgtapose.image_proc.image_from_belief_map(origin_hms_tensor, normalization_method=6)
        hms = np.asarray(origin_hms_img)
        
        assert img.shape == hms.shape # H x W x C
        C, H, W = img.shape
        out = np.zeros((C, H, W))
        print("max", np.max(hms))
        print("min", np.min(hms))
        out[index_full] = hms[index_full] * 0.5 + img[index_full] * 0.5
        out[index_zero] = img[index_zero] * 0.9 + hms[index_zero] * 0.1
        blend_img = Image.fromarray(np.uint8(out))
        img = Image.fromarray(np.uint8(img)) 
        

        detected_kps = detected_kps.tolist()
        gt_kps_raw = []
        trans = get_affine_transform(meta["c"], meta["s"], 0, (meta["inp_width"], meta["inp_height"]), inv=1)
        for idx, dt_kp_raw in enumerate(detected_kps):
            ct_dt = [0,0]
            ct_dt[0] = np.clip(dt_kp_raw[0], 0, meta["width"]-1)
            ct_dt[1] = np.clip(dt_kp_raw[1], 0, meta["height"]-1)
            blend_img = sgtapose.image_proc.overlay_points_on_image(
                        blend_img,
                        [ct_dt],
                        [self.keypoint_names[idx].replace("panda_", "")],
                        annotation_color_dot=["white"],
                        annotation_color_text=["white"],
#                        annotation_color_dot=["green", "red"],
                        point_diameter=4,
                        )
                        
        save_dir = os.path.join(output_dir, str(self.idx).zfill(2))
        self.exists_or_mkdir(save_dir)
        img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_img.png"))
        # hm_whole_mosaic.save(os.path.join(save_dir, f"{str(i).zfill(5)}_heatmap.png"))
        blend_img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_blend_img.png"))
        origin_hms_img.save(os.path.join(save_dir, f"{str(i).zfill(5)}_hms.png"))


        return 
        

    
    def _transform_scale(self, image, scale=1):
      '''
        Prepare input image in different testing modes.
          Currently support: fix short size/ center crop to a fixed size/ 
          keep original resolution but pad to a multiplication of 32
      '''
      # print("image.shape", image.shape)
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      if self.opt.fix_short > 0:
          if height < width:
              inp_height = self.opt.fix_short
              inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
          else:
              inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
              inp_width = self.opt.fix_short
          c = np.array([width / 2, height / 2], dtype=np.float32)
          s = np.array([width, height], dtype=np.float32)
      elif self.opt.fix_res:
          inp_height, inp_width = self.opt.input_h, self.opt.input_w
          c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
          s = max(height, width) * 1.0
          # s = np.array([inp_width, inp_height], dtype=np.float32)
      else:
          inp_height = (new_height | self.opt.pad) + 1
          inp_width = (new_width | self.opt.pad) + 1
          c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
          s = np.array([inp_width, inp_height], dtype=np.float32)
      resized_image = cv2.resize(image, (new_width, new_height))
      return resized_image, c, s, inp_width, inp_height, height, width
  
#    @jit(nopython=True)
    def pre_process(self, image, scale, input_meta={}, i=None):
      '''
      Crop, resize, and normalize image. Gather meta data for post processing 
        and tracking.
      '''
      #t1 = time.time()
      resized_image, c, s, inp_width, inp_height, height, width = self._transform_scale(image)
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
      out_height =  inp_height // self.opt.down_ratio
      out_width =  inp_width // self.opt.down_ratio
      out_height = int(out_height)
      out_width = int(out_width)
      trans_output = get_affine_transform(c, s, 0, [out_width, out_height])
      inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
      inp_image = torch.from_numpy(inp_image)
      inp_image = self.normalize_img(inp_image, self.mean, self.std) 
      images = inp_image.permute(2, 0, 1).view(1, 3, inp_height, inp_width)

      meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
               if 'calib' in input_meta else \
               self._get_default_calib(width, height)}
      meta.update({'c': c, 's': s, 'height': height, 'width': width,
              'out_height': out_height, 'out_width': out_width,
              'inp_height': inp_height, 'inp_width': inp_width,
              'trans_input': trans_input, 'trans_output': trans_output})
      if 'pre_dets' in input_meta:
        meta['pre_dets'] = input_meta['pre_dets']
      if 'cur_dets' in input_meta:
        meta['cur_dets'] = input_meta['cur_dets']
      return images, meta
   
    def normalize_img(self, img, mean, std):
        return ((img / 255. - mean) / std).type(torch.float32)
   
    def _trans_bbox(self, bbox, trans, width, height):
      '''
      Transform bounding boxes according to image crop.
      '''
      bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
      return bbox
  
    def _get_additional_inputs(self, dets, meta, with_hm=True):
      '''
      Render input heatmap from previous trackings.
      '''
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
      repro_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)
      #print(self.opt.num_classes)
      pre_hm_cls = torch.zeros(1, self.opt.num_classes, out_height, out_width)
      repro_hm_cls = torch.zeros(1, self.opt.num_classes, out_height, out_width)

      output_inds = []
      for det in dets:
          if det['score'] < self.opt.pre_thresh or det['active'] == 0:
              continue
          ct_det = affine_transform(det["ct_wreg"], trans_input)
          ct_det[0] = np.clip(ct_det[0], 0, inp_width-1)
          ct_det[1] = np.clip(ct_det[1], 0, inp_height-1)
          ct_det_out = affine_transform(det["ct_wreg"], trans_output)
          ct_det_out[0] = np.clip(ct_det_out[0], 0, out_width-1)
          ct_det_out[1] = np.clip(ct_det_out[1], 0, out_height-1)
          
          radius = 4
          ct = np.array(
          [ct_det[0], ct_det[1]], dtype=np.float32)
          ct_out = np.array(
            [ct_det_out[0], ct_det_out[1]], dtype=np.int32)
          output_inds.append(ct_out[1] * out_width + ct_out[0])
      if with_hm:
          input_hm = input_hm[np.newaxis]
          repro_hm = repro_hm[np.newaxis]
          if self.opt.flip_test:
            input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
          input_hm = torch.from_numpy(input_hm).to(self.opt.device)
          repro_hm = torch.from_numpy(repro_hm).to(self.opt.device)
      output_inds = np.array(output_inds, np.int64).reshape(1, -1)
      output_inds = torch.from_numpy(output_inds).to(self.opt.device)
      return input_hm, repro_hm,  pre_hm_cls.to(self.opt.device), repro_hm_cls.to(self.opt.device),  output_inds

    def _get_further_dt_pnp_inputs(self, kps_detected_raw_np, meta, prev_json, json):
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        
        keypoint_names = self.keypoint_names
        object_name = self.opt.robot
        
        prev_keypoints = sgtapose.utilities.load_seq_keypoints(prev_json, object_name, keypoint_names, self.camera_K)
        next_keypoints = sgtapose.utilities.load_seq_keypoints(json, object_name, keypoint_names, self.camera_K)
        prev_x3d_np = np.array(deepcopy(prev_keypoints["positions_wrt_robot"]))
        next_x3d_np = np.array(deepcopy(next_keypoints["positions_wrt_robot"]))
        n_kp, _ = prev_x3d_np.shape
        
        # print('kps_detected_raw_np', kps_detected_raw_np)
        idx_good_detections = np.where(kps_detected_raw_np > -999.999 * 4)
        idx_good_detections_rows = np.unique(idx_good_detections[0])
        pre_x3d_list = prev_x3d_np[idx_good_detections_rows, :].tolist()
        kps_raw_list = kps_detected_raw_np[idx_good_detections_rows, :].tolist()
        kps_raw_np = np.array(kps_raw_list)
        if kps_raw_list == []:
            # print('empty!')
            return torch.zeros(1, 1, inp_height, inp_width).to(self.opt.device), torch.zeros(1, 1, inp_height, inp_width).to(self.opt.device), \
                    torch.zeros(1, n_kp, out_height, out_width).to(self.opt.device), torch.zeros(1, n_kp, out_height, out_width).to(self.opt.device)
        
        
        prev_kp_projs_dt, next_kp_projs_est = sgtapose.geometric_vision.is_pnp(np.array(pre_x3d_list), np.array(kps_raw_list), next_x3d_np,\
                                              kps_detected_raw_np,  self.camera_K)
        

        pre_hm = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_dt, trans_input, inp_width, inp_height, meta['width'], meta['height'])
        self.pre_hm_teaser = sgtapose.utilities.get_prev_hm_wo_noise_teaser(prev_kp_projs_dt, trans_input, inp_width, inp_height, meta['width'], meta['height'])
        self.pre_hm_teaser = torch.from_numpy(self.pre_hm_teaser).view(1, 1, inp_height, inp_width)
        pre_hm = torch.from_numpy(pre_hm).view(1, 1, inp_height, inp_width)
        repro_hm = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,inp_width, inp_height, meta['width'], meta['height'])
        repro_hm = torch.from_numpy(repro_hm).view(1, 1, inp_height, inp_width)
        
        pre_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_dt, prev_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
        pre_hm_clses = torch.from_numpy(pre_hm_clses).view(1, -1, out_height, out_width)
        repro_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
        repro_hm_clses = torch.from_numpy(repro_hm_clses).view(1, -1, out_height, out_width)
        
        
        return pre_hm.to(self.opt.device), repro_hm.to(self.opt.device), pre_hm_clses.to(self.opt.device), repro_hm_clses.to(self.opt.device)
    
    def _get_further_dt_pnp_inputs_real(self, kps_detected_raw_np, meta, prev_json, json):
        
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']        
        keypoint_names = self.keypoint_names
        
        object_name = "panda"
        # t1 = time.time()
        try:
            prev_keypoints = sgtapose.utilities.load_keypoints(prev_json, object_name, keypoint_names)
            next_keypoints = sgtapose.utilities.load_keypoints(json, object_name, keypoint_names)
            prev_x3d_np = np.array(prev_keypoints["positions_wrt_cam"])
            next_x3d_np = np.array(next_keypoints["positions_wrt_cam"])
        except:
            prev_x3d_np = np.array(sgtapose.utilities.load_depth_x3d(prev_json, keypoint_names))
            next_x3d_np = np.array(sgtapose.utilities.load_depth_x3d(json, keypoint_names))

        
        n_kp, _ = prev_x3d_np.shape
        idx_good_detections = np.where(kps_detected_raw_np > -999.999 * 4)
        idx_good_detections_rows = np.unique(idx_good_detections[0])
        pre_x3d_list = prev_x3d_np[idx_good_detections_rows, :].tolist()
        kps_raw_list = kps_detected_raw_np[idx_good_detections_rows, :].tolist()
        kps_raw_np = np.array(kps_raw_list)
        if kps_raw_list == []:
            # print('empty!')
            return torch.zeros(1, 1, inp_height, inp_width).to(self.opt.device), torch.zeros(1, 1, inp_height, inp_width).to(self.opt.device), \
                    torch.zeros(1, n_kp, out_height, out_width).to(self.opt.device), torch.zeros(1, n_kp, out_height, out_width).to(self.opt.device)
        
        
        prev_kp_projs_dt, next_kp_projs_est = sgtapose.geometric_vision.is_pnp(np.array(pre_x3d_list), np.array(kps_raw_list), next_x3d_np, \
                                            kps_detected_raw_np, self.camera_K)

        pre_hm = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_dt, trans_input, inp_width, inp_height, meta['width'], meta['height'])
        pre_hm = torch.from_numpy(pre_hm).view(1, 1, inp_height, inp_width)
        self.pre_hm_teaser = sgtapose.utilities.get_prev_hm_wo_noise_teaser(prev_kp_projs_dt, trans_input, inp_width, inp_height, meta['width'], meta['height'])
        self.pre_hm_teaser = torch.from_numpy(self.pre_hm_teaser).view(1, 1, inp_height, inp_width)
        repro_hm = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,inp_width, inp_height, meta['width'], meta['height'])
        repro_hm = torch.from_numpy(repro_hm).view(1, 1, inp_height, inp_width)
        
        pre_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_dt, prev_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
        pre_hm_clses = torch.from_numpy(pre_hm_clses).view(1, -1, out_height, out_width)
        repro_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
        repro_hm_clses = torch.from_numpy(repro_hm_clses).view(1, -1, out_height, out_width)
        
        return pre_hm.to(self.opt.device), repro_hm.to(self.opt.device), pre_hm_clses.to(self.opt.device), repro_hm_clses.to(self.opt.device)
        

    def _get_further_dt_inputs(self, kps_detected_raw_np, meta,):
      '''
      Render input heatmap from previous trackings.
      '''
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
      idx_good_detections = np.where(kps_detected_raw_np > -999.999 * 4)
      idx_good_detections_rows = np.unique(idx_good_detections[0])
      kps_raw_list = kps_detected_raw_np[idx_good_detections_rows, :].tolist()
      if kps_raw_list == []:
          # print('empty!')
          return torch.zeros(1, 1, inp_height, inp_width).to(self.opt.device)
      kps_raw_np = np.array(kps_raw_list)
      pre_hm = sgtapose.utilities.get_prev_hm_wo_noise(kps_raw_np, trans_input, inp_width, inp_height, meta['width'], meta['height'])
      # pre_hm = sgtapose.utilities.get_prev_hm_wo_noise_old(kps_raw_np, trans_input, inp_width, inp_height)
      pre_hm = torch.from_numpy(pre_hm).view(1, 1, inp_height, inp_width)
      return pre_hm.to(self.opt.device)
          
    
    def _get_initial_gt_inputs(self, prev_json, json, meta, with_hm=True):
      '''
      Render input heatmap from previous trackings.
      '''
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
      output_inds = []
      parser = YAML(typ="safe")
      keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
      object_name = "panda"
        
      prev_keypoints = sgtapose.utilities.load_keypoints(prev_json, object_name, keypoint_names)
      next_keypoints = sgtapose.utilities.load_keypoints(json, object_name, keypoint_names)
      prev_x3d_np = np.array(prev_keypoints["positions_wrt_cam"])
      next_x3d_np = np.array(next_keypoints["positions_wrt_cam"])      
      gt_kps_raw_np = np.array(prev_keypoints["projections"])
      
      
      pnp_retval, next_kp_projs_est, prev_kp_projs_noised_np = sgtapose.geometric_vision.get_pnp_keypoints(prev_x3d_np, gt_kps_raw_np, next_x3d_np, self.camera_K, self.opt.hm_disturb, self.opt.lost_disturb) 
        
      pre_hm = sgtapose.utilities.get_prev_hm_wo_noise(prev_kp_projs_noised_np, trans_input, inp_width, inp_height, meta['width'], meta['height'])
      pre_hm = torch.from_numpy(pre_hm).view(1, 1, inp_height, inp_width)
          
      repro_hm = sgtapose.utilities.get_prev_hm_wo_noise(next_kp_projs_est, trans_input,inp_width, inp_height, meta['width'], meta['height'])
      repro_hm = torch.from_numpy(repro_hm).view(1, 1, inp_height, inp_width)
      
      
      pre_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(prev_kp_projs_noised_np, prev_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
      pre_hm_clses = torch.from_numpy(pre_hm_clses).view(1, -1, out_height, out_width)
      repro_hm_clses = sgtapose.utilities.get_prev_hm_wo_noise_cls(next_kp_projs_est, next_x3d_np, trans_output, out_width, out_height, meta['width'], meta['height'])
      repro_hm_clses = torch.from_numpy(repro_hm_clses).view(1, -1, out_height, out_width)
      
      return pre_hm.to(self.opt.device), repro_hm.to(self.opt.device), pre_hm_clses.to(self.opt.device), repro_hm_clses.to(self.opt.device)
          

    def _get_final_kps(self, dets):
      '''
      Render input heatmap from previous trackings.
      '''
      dets = deepcopy(dets)
      detected_kps = np.full((self.opt.num_classes, 2), -999.999 * 4) 
      
      cls = {}
      for i in range(1, self.opt.num_classes + 1):
          cls[i] = {"x" : [], "y":[]}
      for det in dets:
         
          if self.is_ct:
              score, clas, ct_wreg = det["score"], det["class"], det["ct_wreg"].tolist()
          else:
              score, clas, ct_wreg = det["score"], det["class"], det["ct"].tolist()
          
          cls[clas]["x"].append([score, ct_wreg[0]])
          cls[clas]["y"].append([score, ct_wreg[1]]) 
      
      for i in range(1, self.opt.num_classes+1):
          try:
              x_list, y_list = cls[i]["x"], cls[i]["y"]
              x_list.sort()
              y_list.sort()
              assert x_list[-1][0] == y_list[-1][0]
              x_res = x_list[-1][1]
              y_res = y_list[-1][1]
              detected_kps[i-1] = [x_res, y_res]
              # print('res', [x_res, y_res])
          except:
              pass
      
      if not self.is_ct:
          img = Image.open(self.pre_img_path).convert("RGB")
          
          dt_kps_raw = sgtapose.image_proc.convert_keypoints_to_raw_from_netin(
                       detected_kps, 
                       (self.opt.input_w, self.opt.input_h),
                        img.size,
                        "shrink-and-crop")
          detected_kps = dt_kps_raw
                  
      return detected_kps
      

    def _get_final_kps_output(self, dets):
      '''
      Render input heatmap from previous trackings.
      '''
      dets = deepcopy(dets)
      detected_kps = np.full((self.opt.num_classes, 6), -999.999 * 4) 
      cls = {}
      for i in range(1, self.opt.num_classes + 1):
          cls[i] = {"x" : [], "y":[]}
      for det in dets:
          score, clas, ct_wreg, ct, reg = det["score"], det["class"], det["ct_wreg_output"].tolist(), det["ct_output"].tolist(), det["reg_output"].tolist()
          cls[clas]["x"].append([score, ct_wreg[0], ct[0], reg[0]])
          cls[clas]["y"].append([score, ct_wreg[1], ct[1], reg[1]])
      
      for i in range(1, self.opt.num_classes+1):
          try:
              x_list, y_list = cls[i]["x"], cls[i]["y"]
              x_list.sort()
              y_list.sort()
              assert x_list[-1][0] == y_list[-1][0]
              x_res, x_ct, x_offset = x_list[-1][1:]
              y_res, y_ct, y_offset = y_list[-1][1:]
              detected_kps[i-1] = [x_res, y_res, x_ct, y_ct, x_offset, y_offset]
          except:
              pass
                  
      return detected_kps
          
   
    def _get_default_calib(self, width, height):
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                          [0, self.rest_focal_length, height / 2, 0], 
                          [0, 0, 1, 0]])
      return calib
    
    def _get_heatmap_and_topk(self, topk_indices, pre_hm, save_dir, i):
        pre_hm_save = pre_hm[0][0]
        pre_hm_img = sgtapose.image_proc.image_from_belief_map(pre_hm_save)
        overlay_whole_image = sgtapose.image_proc.overlay_points_on_image(
                self.pre_image,
                topk_indices,
                point_diameter=2,
            )
        pre_hm_img.save(save_dir + f"{str(i).zfill(5)}_prehm.png")
        overlay_whole_image.save(save_dir + f"{str(i).zfill(5)}_topkimg.png")

        
        
        
    
    def _get_overlay_imgs(self, hms, img, detected_kps_raw_np, json_path, meta, output_dirs, i):
         
        parser = YAML(typ="safe")
        with open(json_path, "r") as f:
            data = parser.load(f.read().replace('\t', ' '))
        # print(data)
        data = data["objects"][0]
        object_keypoints = data["keypoints"]
        gt_kps_raw = []
        detected_kps_raw = detected_kps_raw_np.tolist()
        overlay_whole_images = []
        overlay_gt_images = []
        overlay_dt_images = [] 
        overlay_blend_images = []
        
        
        
        hms_imgs = []
        trans = get_affine_transform(meta["c"], meta["s"], 0, (meta["out_width"], meta["out_height"]), inv=1)
        for j in range(hms.shape[0]):
            out_hm = cv2.warpAffine(
            hms[j].cpu().numpy(), trans, (meta["width"], meta["height"]),flags=cv2.INTER_LINEAR)
            out_hm_tensor = torch.from_numpy(out_hm).to(self.opt.device)
            out_hm_img = sgtapose.image_proc.image_from_belief_map(out_hm_tensor)
            # print(out_hm_img.size)
            hms_imgs.append(out_hm_img)
            
        
        for idx, kp_name in enumerate(self.keypoint_names):
            gt_kps_raw.append(object_keypoints[idx]["projected_location"])
        
        
        for idx, (gt_kp_raw, detected_kp_raw) in enumerate(zip(gt_kps_raw, detected_kps_raw)):
            #print('i', i)
            ct_gt = gt_kp_raw
            ct_gt[0] = np.clip(ct_gt[0], 0, meta["width"]-1)
            ct_gt[1] = np.clip(ct_gt[1], 0, meta["height"]-1)
            
            ct_dt = detected_kp_raw
            ct_dt[0] = np.clip(ct_dt[0], 0, meta["width"]-1)
            ct_dt[1] = np.clip(ct_dt[1], 0, meta["height"]-1)
            
            blend_image = Image.blend(
                hms_imgs[idx], img, alpha=0.5
            )
            
            overlay_whole_image = sgtapose.image_proc.overlay_points_on_image(
                blend_image,
                [ct_gt, ct_dt],
                [self.keypoint_names[idx]] * 2,
                annotation_color_dot=["green", "red"],
                annotation_color_text=["green", "red"],
#                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )
            overlay_whole_images.append(overlay_whole_image)
            overlay_blend_images.append(blend_image)
            
#            
        overlay_whole_mosaic = sgtapose.image_proc.mosaic_images(
            overlay_whole_images, rows=2, cols=4, inner_padding_px=10
        )
        overlay_blend_mosaic = sgtapose.image_proc.mosaic_images(
            overlay_blend_images, rows=2, cols=4, inner_padding_px=10
        )

        save_dir = os.path.join(output_dirs["whole"], str(self.idx).zfill(2))
        self.exists_or_mkdir(save_dir)
        overlay_whole_mosaic.save(os.path.join(save_dir, f"{str(i).zfill(5)}_overlay.png"))
        overlay_blend_mosaic.save(os.path.join(save_dir, f"{str(i).zfill(5)}_blend.png"))

    
    def _get_overlay_output_imgs(self, detected_kps_output_np, meta,json_path, output_dirs, i):
        # print('output_kps', detected_kps_output_np)
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        
        parser = YAML(typ="safe")
        with open(json_path, "r") as f:
            data = parser.load(f.read().replace('\t', ' '))
        data = data[0]
        object_keypoints = data["keypoints"]
        # gt_kps_output = []    
        gt_kps_raw = [] 
        for idx, kp_name in enumerate(self.keypoint_names):
            gt_kps_raw.append(object_keypoints[idx]["projected_location"])
        
        gt_kps_raw = sgtapose.image_proc.convert_keypoints_to_netin_from_raw(
        gt_kps_raw, (640, 360), (self.opt.input_w, self.opt.input_h), 'shrink-and-crop')
        
        gt_kps_raw_np = np.stack(gt_kps_raw)
        gt_kps_output_np = sgtapose.utilities.affine_transform_and_clip(gt_kps_raw_np, trans_output, out_width, out_height)
        gt_kps_output = gt_kps_output_np.tolist()
        
        gt_kps_output_nps = sgtapose.utilities.affine_transforms(gt_kps_raw_np, trans_output, out_width, out_height)
        
        next_hms = sgtapose.utilities.get_hm(gt_kps_output_np, out_width, out_height)
        next_hms_res = []
        for j in range(next_hms.shape[0]):
            c, r = np.where(next_hms[j] == np.max(next_hms[j]))
            next_hms_res.append([r[0], c[0]])
    
        output_tensor = torch.ones(3, out_height, out_width).to(self.opt.device)
        output_img = sgtapose.image_proc.image_from_tensor(output_tensor / 255)
        detected_kps_output = detected_kps_output_np[:, :2].tolist()
        overlay_dt_images = []
        for idx, (detected_kp_output, gt_kp_output) in enumerate(zip(detected_kps_output, gt_kps_output)):
            ct_dt = detected_kp_output
            ct_gt = gt_kp_output
            # print('i', i)
            
            overlay_dt_image = sgtapose.image_proc.overlay_points_on_image(
                output_img,
                [ct_gt, ct_dt],
                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )

            dt_dir = output_dirs['dt']
            this_dt_dir = os.path.join(dt_dir, f"{idx}_kps_output", "")
            # print(this_whole_dir)
            self.exists_or_mkdir(this_dt_dir)
            s = str(i).zfill(5)
            overlay_dt_image.save(this_dt_dir + f"{s}.png")
            if i == 29:
                video_list = os.listdir(this_dt_dir)
                video_list.sort()
                image = Image.open(this_dt_dir + video_list[0])
                video = cv2.VideoWriter(this_dt_dir + "dt.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2, image.size)
                for j in range(1, len(video_list) + 1):
                    img= cv2.imread(os.path.join(this_dt_dir, video_list[j-1])) # 读取图片
                    video.write(img)
                video.release()
        
        n_kp = gt_kps_output_np.shape[0]
        kps_output_save = np.ones((n_kp * 3, 3))
        kps_output_save[:n_kp, :2] = gt_kps_output_nps
        kps_output_save[n_kp:2*n_kp, :2] = detected_kps_output_np[:, 2:4]
        kps_output_save[2*n_kp:, :2] = detected_kps_output_np[:, 4:]
        txt_dir = os.path.join(output_dirs['dt'], "doc", "")
        self.exists_or_mkdir(txt_dir)
        s = str(i).zfill(5)
        txt_dir = txt_dir + f"{s}.txt"
        np.savetxt(txt_dir, kps_output_save)
        
        
        
        
    
    def _sigmoid_output(self, output):
      if 'hm' in output:
          output['hm'] = output['hm'].sigmoid_()
      if 'hm_hp' in output:
          output['hm_hp'] = output['hm_hp'].sigmoid_()
      if 'dep' in output:
          output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
          output['dep'] *= self.opt.depth_scale
      return output

    def _flip_output(self, output):
        average_flips = ['hm', 'wh', 'dep', 'dim']
        neg_average_flips = ['amodel_offset']
        single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
        'hp_offset', 'rot', 'tracking', 'pre_hm']
        for head in output:
            if head in average_flips:
                output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
            if head in neg_average_flips:
                flipped_tensor = flip_tensor(output[head][1:2])
                flipped_tensor[:, 0::2] *= -1
                output[head] = (output[head][0:1] + flipped_tensor) / 2
            if head in single_flips:
                output[head] = output[head][0:1]

        return output
      
    def process(self, images, pre_images=None, pre_hms=None, repro_hms=None,
      pre_inds=None, return_time=False, pre_hms_cls=None, repro_hms_cls=None):
      with torch.no_grad():
        torch.cuda.synchronize()
        if self.phase == "CenterTrack+Repro":
            output = self.model(images, pre_images, repro_hms)[-1]
        elif self.phase == "Dream":
            output = self.model(images)[-1]
            # print("output.shape", output["hm"].shape)
        elif self.phase == "PlanA":
            output = self.model(images, pre_images, pre_hms, repro_hms)[-1]
            # print("next hm's shape", output["hm"].shape)
        elif self.phase == "CenterTrack":
            output = self.model(images, pre_images, pre_hms)[-1]
        elif self.phase == "CenterNet":
            output = self.model(images)[-1]
        elif self.phase == "CenterTrack-Pre_hm":
            output = self.model(images, pre_images)[-1]
        elif self.phase == "PlanA_win":
            output = self.model(images, pre_images, pre_hms, repro_hms, pre_hms_cls, repro_hms_cls)[-1]
        elif self.phase == "PlanACAT":
            output = self.model(images, pre_images, pre_hms)[-1]
        elif self.phase == "ablation_wo_shared" or self.phase == "ablation_shared":
            output = self.model(images, pre_images, pre_hms)[-1]
        elif self.phase == "ablation_shared_repro":
            output = self.model(images, pre_images, pre_hms, repro_hms)[-1]
        else:
            raise ValueError
        # output = self.model(images)[-1]
        if self.is_ct:
            output = self._sigmoid_output(output)
        output.update({'pre_inds': pre_inds})
        if self.opt.flip_test:
          output = self._flip_output(output)
        torch.cuda.synchronize()
        forward_time = time.time()
 
        
        dets = dream_generic_decode(output, K=self.opt.num_classes, opt=self.opt)
        hms = output["hm"][0]
        torch.cuda.synchronize()
        for k in dets:
          dets[k] = dets[k].detach().cpu().numpy()
      if return_time:
        return output, dets, forward_time, hms
      else:
        return output, dets
    
    def post_process(self, dets, meta, scale=1):
      dets = dream_generic_post_process(
        self.opt, dets, [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes,
        [meta['calib']], meta['height'], meta['width'])
      self.this_calib = meta['calib']
      
      if scale != 1:
        for i in range(len(dets[0])):
          for k in ['bbox', 'hps']:
            if k in dets[0][i]:
              dets[0][i][k] = (np.array(
                dets[0][i][k], np.float32) / scale).tolist()
      return dets[0]
    
    def transform_topk_coords_to_input(self, topk_indices, meta):
        inp_height, inp_width = meta["inp_height"], meta["inp_width"]
        _, K = topk_indices.shape
        topk_coords = torch.zeros(K, 2) # x, y
        topk_coords[:, 0] = topk_indices[0] % inp_width
        topk_coords[:, 1] = topk_indices[0] // inp_width
        trans = get_affine_transform(
      meta['c'], meta['s'], 0, (inp_width, inp_height), inv=1).astype(np.float32)
        input_K = transform_preds_with_trans(topk_coords, trans)
        return input_K # K x 2
    
    def merge_outputs(self, detections):
      assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
      results = []
      for i in range(len(detections[0])):
        if detections[0][i]['score'] > self.opt.out_thresh:
          results.append(detections[0][i])
      return results
  
    def debug(self, debugger, images, dets, output, scale=1, 
      pre_images=None, pre_hms=None):
      img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      if 'hm_hp' in output:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')

      if pre_images is not None:
        pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img')
        if pre_hms is not None:
          pre_hm = debugger.gen_colormap(
            pre_hms[0].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')


    def show_results(self, debugger, image, results):
      debugger.add_img(image, img_id='generic')
      if self.opt.tracking:
        debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
          img_id='previous')
        self.pre_image_ori = image
      
      for j in range(len(results)):
        if results[j]['score'] > self.opt.vis_thresh:
          if 'active' in results[j] and results[j]['active'] == 0:
            continue
          item = results[j]
          if ('bbox' in item):
            sc = item['score'] if self.opt.demo == '' or \
              not ('tracking_id' in item) else item['tracking_id']
            sc = item['tracking_id'] if self.opt.show_track_color else sc
            
            debugger.add_coco_bbox(
              item['bbox'], item['class'] - 1, sc, img_id='generic')

          if 'tracking' in item:
            debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
          
          tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
          if 'tracking_id' in item and self.opt.demo == '' and \
            not self.opt.show_track_color:
            debugger.add_tracking_id(
              item['ct'], item['tracking_id'], img_id='generic')

          if (item['class'] in [1, 2]) and 'hps' in item:
            debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
              img_id='generic')

      if len(results) > 0 and \
        'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
        debugger.add_3d_detection(
          image if not self.opt.qualitative else cv2.resize(
            debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])), 
          False, results, self.this_calib,
          vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
        debugger.add_bird_view(
          results, vis_thresh=self.opt.vis_thresh,
          img_id='bird_pred', cnt=self.cnt)
        if self.opt.show_track_color and self.opt.debug == 4:
          del debugger.imgs['generic'], debugger.imgs['bird_pred']
      if 'ddd_pred' in debugger.imgs:
        debugger.imgs['generic'] = debugger.imgs['ddd_pred']
      if self.opt.debug == 4:
        debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
      else:
        debugger.show_all_imgs(pause=self.pause)
    

    def reset_tracking(self):
      self.tracker.reset()
      self.pre_images = None
      self.pre_image_ori = None
    
    def exists_or_mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        else:
            return True
