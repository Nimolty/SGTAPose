# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:52:39 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tools._init_paths as _init_paths

import os
import sys
import cv2
import json
import copy
import glob
import numpy as np
from lib.opts_parallel import opts
from lib.sgta_detector import SGTADetector
import torch
from tqdm import tqdm
import sgtapose
from ruamel.yaml import YAML
from PIL import Image as PILImage
import time
import random
from tqdm import tqdm


def find_dataset(opt):
    keypoint_names = opts().get_keypoint_names(opt)
    input_dir = opt.infer_dataset
    input_dir = os.path.expanduser(input_dir) # 
    assert os.path.exists(input_dir),\
    'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir) # 
    print(dirlist)
    dirlist.sort()
    
    found_videos = []
    for each_dir in dirlist:
        if each_dir.endswith(".json"):
            continue
        output_dir = os.path.join(input_dir, each_dir)
        # output_dir = ../../franka_data_0825/xxxxx
        found_video = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                       if f.endswith('_color.png')]
        found_json = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                        if f.endswith("meta.json")]
        found_video.sort()
        found_json.sort()
        if len(found_video) != len(found_json):
            continue
        found_videos.append([found_video, found_json])
    
    return found_videos

def inference(opt):
    keypoint_names = opts().get_keypoint_names(opt)
    print("inference dataset", opt.infer_dataset)
    with torch.no_grad():
        found_videos = find_dataset(opt)
        json_list, detected_kps_list = [], []
        print("length of found_videos", len(found_videos))
        
        
        
        for video_idx, found_video_0 in tqdm(enumerate(found_videos[:50])):
            # print(found_video_0) 
            length = len(found_video_0[0]) 
#            if video_idx >= 0:
#                continue            
            
            detector = SGTADetector(opt, keypoint_names, is_real=False, is_ct=opt.is_ct, idx=video_idx)
            for i, img_path in enumerate(found_video_0[0]):
#                if i == 0:
#                    continue
                json_path = found_video_0[1][i]
#                print('img_path', img_path)
#                print('json_path', json_path)
                img = cv2.imread(img_path)
#                raw_height, raw_width, _ = img.shape #720, 1280, 3
#                img = cv2.resize(img, (raw_width // 2, raw_height // 2))
                
                if not opt.is_ct:
                    img = PILImage.open(img_path).convert("RGB")
                    # print("#################### SIZE ####################")
                    # print('size', img.size)
                    # print("#################### SIZE ####################")
                    img_shrink_and_crop = sgtapose.image_proc.preprocess_image(
                    img, (opt.input_w, opt.input_h), "shrink-and-crop"
                    )
                    img = np.asarray(img_shrink_and_crop)
                    # print("img.shape", img.shape)
                
                t1 = time.time()
                ret, detected_kps_np, _ = detector.run(img, i, json_path, is_final=True, teaser_flag=False, img_path=img_path)  
                # print("detected_kps_np", detected_kps_np)
                t2 = time.time()
#                print("", t2 - t1)          
                output_dir = img_path.rstrip('png')
                # np.savetxt(output_dir + 'txt', detected_kps_np)
                json_list.append(json_path)
                detected_kps_list.append(detected_kps_np) 
                    # print(detected_kps)
    
    if opt.is_ct:
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        sgtapose.utilities.exists_or_mkdir(output_dir)
#        
        save_name = opt.infer_dataset.split('/')[-2]
        path_meta = os.path.join(output_dir, f"{save_name}_dt_and_json.json")
        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(detected_kps_list).tolist()
            meta_json["json"] = json_list
        #
            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()
#        
        
        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            detected_kps_list = real_data["dt"]
            json_list = real_data["json"] 
        
        
        analysis_info = sgtapose.analysis.analyze_ndds_center_dream_dataset(
        json_list, # 
        detected_kps_list,
        opt, 
        keypoint_names,
        [640, 360],
        output_dir,
        is_real=False,
        set_mode=save_name)
        return analysis_info
    else:
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        sgtapose.utilities.exists_or_mkdir(output_dir)
        save_name = opt.infer_dataset.split('/')[-2]
        path_meta = os.path.join(output_dir, f"{save_name}_dt_and_json.json")
        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(detected_kps_list).tolist()
            meta_json["json"] = json_list
    
            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()
    
        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            detected_kps_list = real_data["dt"]
            json_list = real_data["json"] 
        
        analysis_info = sgtapose.analysis.analyze_ndds_center_dream_dataset(
        json_list, # 
        detected_kps_list,
        opt, 
        keypoint_names,
        [640, 360],
        output_dir,
        is_real=False,
        set_mode=save_name)
        return analysis_info
        



def inference_real(opt):
    real_info_path = opt.real_info_path
    real_info_path = os.path.join(real_info_path, opt.is_real + "_split_info.json")
    real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    parser = YAML(typ="safe")
    print(real_info_path)

    with open(real_info_path, "r") as f:
        real_data = parser.load(f.read().replace('\t', ' '))
        real_jsons = real_data["json_paths"]
        real_images = real_data["img_paths"] # 
    json_list, detected_kps_list, d_lst = [], [], []
    json_lists, detected_kps_lists = [], []
    count = 0
    with torch.no_grad():
        for idx, (video_images, video_jsons) in tqdm(enumerate((zip(real_images, real_jsons)))):
            this_json_list, this_detected_kp_proj_list = [], []
                
            count = idx
            detector = SGTADetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=opt.is_ct, idx=idx)
            assert len(video_images) == len(video_jsons)
            length = len(video_images)
            
            for j, (img_path, json_path) in tqdm(enumerate(zip(video_images, video_jsons))):
                img_path = os.path.join(opt.infer_dataset, opt.is_real, img_path)
                json_path = os.path.join(opt.infer_dataset, opt.is_real, json_path)

                img = cv2.imread(img_path)
                raw_height, raw_width, _ = img.shape 
                if not opt.is_ct:
                    img = PILImage.open(img_path).convert("RGB")
                    img_shrink_and_crop = sgtapose.image_proc.preprocess_image(
                    img, (opt.input_w, opt.input_h), "shrink-and-crop"
                    )
                    img = np.asarray(img_shrink_and_crop)

                ret, detected_kps_np, _ = detector.run(img, j, json_path, is_final=True, img_path=img_path)
                output_dir = img_path.rstrip('png')
                np.savetxt(output_dir + 'txt', detected_kps_np)
                json_list.append(json_path)
                detected_kps_list.append(detected_kps_np) 
                this_json_list.append(json_path)
                this_detected_kp_proj_list.append(detected_kps_np.tolist())

            json_lists.append(this_json_list)
            detected_kps_lists.append(this_detected_kp_proj_list)
        
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        print("output_dir, output_dir")

        sgtapose.utilities.exists_or_mkdir(output_dir)
        

        if opt.is_real == "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json.json")
        elif opt.is_real == "panda-3cam_realsense" and opt.multi_frame != 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_multi.json")
        elif opt.is_real != "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}.json")
        else:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}_multi.json")
        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(detected_kps_list).tolist()
            meta_json["json"] = json_list
            if opt.multi_frame > 0:
                meta_json["dt_multi"] = detected_kps_lists
                meta_json["json_multi"] = json_lists

            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()

        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            detected_kps_list = real_data["dt"]
            json_list = real_data["json"] 
            if opt.multi_frame > 0:
                detected_kp_proj_lists = real_data["dt_multi"]
                json_lists = real_data["json_multi"]


        if opt.multi_frame == 0:
            analysis_info = sgtapose.analysis.analyze_ndds_center_dream_dataset(
            json_list, # 
            detected_kps_list,
            opt, 
            real_keypoint_names,
            [640, 480],
            output_dir,
            is_real=opt.is_real)
            return analysis_info
        else:
            sgtapose.analysis.solve_multiframe_pnp(
            json_lists,
            detected_kp_proj_lists,
            opt,
            real_keypoint_names,
            [raw_width, raw_height],
            output_dir,
            multiframe=opt.multi_frame,
            is_real=opt.is_real,
            )


def inference_real_multiframe(opt):
    random.seed(opt.seed)
    print("opt.seed", opt.seed)
    #real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    real_keypoint_names = ["Link0","Link1","Link3","Link4", "Link6","Link7","panda_hand",]
    root_image_path = f"/root/autodl-tmp/camera_to_robot_pose/Dream_ty/{opt.note}/color/"
    root_json_path = f"/root/autodl-tmp/camera_to_robot_pose/Dream_ty/{opt.note}/meta_ty/"
    real_jsons = glob.glob(os.path.join(root_json_path, '*.json'))
    real_images = glob.glob(os.path.join(root_image_path, '*.png'))
    real_jsons.sort()
    real_images.sort()
    
    assert len(real_jsons) == len(real_images)
    
    meta_loc = {
    "real_1107" : [
    "000106", "000233", "000360", "000486",
    "000612", "000741", "000868", "000995",
    "001124", "001254", "001380", "001509",
    "001633", "001763", "001897", "002027",
    "002143", "002255", "002373", "002498",
    "002623", "002743", "002862", "002981",
    "003102", "003226", "003351", "003472",
    "003595", "003713", "003840"
    ],    
    "real_1107_with_background" : [
    "000108", "000229", "000353", "000483",
    "000611", "000736", "000861", "000984",
    "001106", "001227", "001352", "001479",
    "001606", "001729", "001857", "001984",
    "002108", "002232", "002351", "002474",
    "002596", "002722", "002850", "002975",
    "003100", "003225", "003351", "003474",
    "003598", "003716", "003836"
    ],
    "real_1108" : \
    ['000232', '000481', '001214', '001335', '001455', '001695', '002059', '002421', '002540', '002656', '002894', '003134', '003733', '003854', '003971', '004088', '004323', '004566', '004693', '004812'],
    "real_1107_with_background" : ["000111"]
    }
    compare_lst = meta_loc[opt.note]
    draw_lst = [5, 751, 1293]
    draw_lst2 = range(1120, 1201)
    draw_lst += draw_lst2
    print("draw_lst", draw_lst)

    # json_list, detected_kps_list, d_lst = [], [], []
    json_list, dt_kps_projs_list = [], []
    count = 0
    with torch.no_grad():
        # detector = SGTADetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=True, idx=0)
        detector = SGTADetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=opt.is_ct, idx=0)
        for j, (img_path, json_path) in tqdm(enumerate(zip(real_images, real_jsons))):
            img_idx = img_path.split('/')[-1].replace(".png", "")
            json_idx = json_path.split('/')[-1].replace(".json", "")
            assert img_idx == json_idx
            # if j == 0 or j >= 30:
            if j == 0:
                continue

            img = cv2.imread(img_path) 
            if not opt.is_ct:
                img = PILImage.open(img_path).convert("RGB")
                img_shrink_and_crop = sgtapose.image_proc.preprocess_image(
                img, (opt.input_w, opt.input_h), "shrink-and-crop"
                )
                img = np.asarray(img_shrink_and_crop)
 
 
            save_dir = f"/root/autodl-tmp/camera_to_robot_pose/topk_check/0/"
            sgtapose.utilities.exists_or_mkdir(save_dir)

            if compare_lst[count] in json_idx:
                ret, detected_kps_np, camera_K = detector.run(img, j, json_path, is_final=True, save_dir=save_dir,teaser_flag=False,img_path=img_path)
                json_list.append(json_path)
                dt_kps_projs_list.append(detected_kps_np.tolist()) 
            else:
                ret, detected_kps_np, camera_K = detector.run(img, j, json_path, is_final=True, save_dir=save_dir,teaser_flag=False,img_path=img_path)
        
        
        gt_kps_pos_list = [sgtapose.utilities.load_depth_x3d(json_path, real_keypoint_names) for json_path in json_list]
        
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        sgtapose.utilities.exists_or_mkdir(output_dir)
       

        path_meta = os.path.join(output_dir, f"dt_and_json_{count}.json")

        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(dt_kps_projs_list).tolist()
            meta_json["gt"] = gt_kps_pos_list
            meta_json["camera_K"] = camera_K.tolist()
           
            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()

        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            dt_kps_projs_list = real_data["dt"]
            gt_kps_pos_list = real_data["gt"] 
            camera_K = np.array(real_data["camera_K"])
            

def inference_real_depth(opt):
    real_path = opt.depth_dataset
    real_images = glob.glob(os.path.join(real_path, opt.is_real, "*png"))
    real_jsons = [i.replace("png", "json") for i in real_images]
    
    real_images = [real_images]
    real_jsons = [real_jsons]
    json_list, detected_kps_list, d_lst = [], [], []
    json_lists, detected_kps_lists = [], []
    count = 0    
    keypoint_names = opts().get_keypoint_names(opt)


    with torch.no_grad():
        for idx, (video_images, video_jsons) in tqdm(enumerate((zip(real_images, real_jsons)))):
            this_json_list, this_detected_kp_proj_list = [], []
            if idx >= 0:  
                pass
                count = idx
                detector = SGTADetector(opt,keypoint_names, is_real=opt.is_real, is_ct=opt.is_ct, idx=idx)
                assert len(video_images) == len(video_jsons)
                length = len(video_images)
                
                for j, (img_path, json_path) in tqdm(enumerate(zip(video_images, video_jsons))):
                    if j >= 0:
                        pass
                    
                    img = cv2.imread(img_path)
                    raw_height, raw_width, _ = img.shape #480, 640, 3
                    # print('img.shape', img.shape)
                    
                    if not opt.is_ct:
                        img = PILImage.open(img_path).convert("RGB")
                        # print("#################### SIZE ####################")
                        # print('size', img.size)
                        # print("#################### SIZE ####################")
                        img_shrink_and_crop = sgtapose.image_proc.preprocess_image(
                        img, (opt.input_w, opt.input_h), "shrink-and-crop"
                        )
                        img = np.asarray(img_shrink_and_crop)

                    # t1 = time.time()
                    ret, detected_kps_np, _ = detector.run(img, j, json_path, is_final=True, img_path=img_path)
                    output_dir = img_path.rstrip('png')
                    np.savetxt(output_dir + 'txt', detected_kps_np)
                    json_list.append(json_path)
                    detected_kps_list.append(detected_kps_np) 
                    this_json_list.append(json_path)
                    this_detected_kp_proj_list.append(detected_kps_np.tolist())

            json_lists.append(this_json_list)
            detected_kps_lists.append(this_detected_kp_proj_list)
                    # print("shape", detected_kps_np.shape)
                    # d_lst.append(detected_kps_np.tolist())
        
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        sgtapose.utilities.exists_or_mkdir(output_dir)
        

        # path_meta = os.path.join(output_dir, f"dt_and_json_{count}.json")
        if opt.is_real == "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json.json")
        elif opt.is_real == "panda-3cam_realsense" and opt.multi_frame != 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_multi.json")
        elif opt.is_real != "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}.json")
        else:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}_multi.json")
        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(detected_kps_list).tolist()
            meta_json["json"] = json_list
            if opt.multi_frame > 0:
                meta_json["dt_multi"] = detected_kps_lists
                meta_json["json_multi"] = json_lists

            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()
 
        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            detected_kps_list = real_data["dt"]
            json_list = real_data["json"] 
            if opt.multi_frame > 0:
                detected_kp_proj_lists = real_data["dt_multi"]
                json_lists = real_data["json_multi"]

        if opt.kps_name == "dream_7":
            analysis_info = sgtapose.analysis.analyze_ndds_center_dream_dataset(
                json_list, # 
                detected_kps_list,
                opt, 
                keypoint_names,
                [640, 360],
                output_dir,
                is_real=opt.is_real,
                dataset_path=real_path)
            return analysis_info
        elif opt.kps_name == "ours_42":
            analysis_info = sgtapose.analysis.analyze_ndds_center_dream_ours_42_dataset(
                json_list, # 
                detected_kps_list,
                opt, 
                keypoint_names,
                [640, 360],
                output_dir,
                is_real=opt.is_real,
                dataset_path=real_path)
            return analysis_info            
              
    
     

if __name__ == "__main__":
    opt = opts().init_infer(7, (480, 480))
    inference_real(opt)

    
























