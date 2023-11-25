# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import random
import torch.nn.functional as F
import numpy as np
from ruamel.yaml import YAML
import torch
import cv2
import math
from copy import deepcopy
import json

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_x3d(json_path, keypoint_names):
    json_in = open(json_path, 'r')
    json_data = json.load(json_in)
    json_data = json_data["objects"][0]["keypoints"]
    x3d_list = []
    for idx, kp_name in enumerate(keypoint_names):
        data = json_data[idx]
        assert kp_name == data["name"]
        x3d_wrt_cam = data["location"]
        x3d_list.append(x3d_wrt_cam)
    return x3d_list

def load_depth_x3d(json_path, keypoint_names):
    json_in = open(json_path, 'r')
    keypoints_data = json.load(json_in)[0]["keypoints"]
    x3d_list = []
    count = 0
    for idx, kp_name in enumerate(keypoint_names):
        while keypoints_data[count]["Name"] != kp_name:
            count += 1
        assert (keypoints_data[count]["Name"] == kp_name), \
            "Expected keypoint '{}' to exist in the datafile '{}', but it does not. \
            Rather, the keypoints are '{}'".format(
            kp_name, json_path, keypoints_data[count]['Name']
            )
        x3d_wrt_rob = keypoints_data[count]["location_wrt_cam"]
        x3d_list.append(x3d_wrt_rob)
    return x3d_list

def load_depth_joints_x3d(json_path):
    json_in = open(json_path, 'r')
    joints_data = json.load(json_in)[0]["joints_3n_fixed_42"]
    x3d_list = [i["location_wrt_cam"] for i in joints_data]
    return x3d_list

def load_x3d_rob(json_path, keypoint_names):
    json_in = open(json_path, 'r')
    json_data = json.load(json_in)
    json_data = json_data["objects"][0]["keypoints"]
    x3d_list = []
    for idx, kp_name in enumerate(keypoint_names):
        data = json_data[idx]
        assert kp_name == data["name"]
        x3d_wrt_cam = data["location_wrt_rob"]
        x3d_list.append(x3d_wrt_cam)
    return x3d_list
def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))

def set_random_seed(seed):
    assert isinstance(
        seed, int
    ), 'Expected "seed" to be an integer, but it is "{}".'.format(type(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def makedirs(directory, exist_ok=False):
    """A method that replicates the functionality of os.makedirs that works for both Python 2 and Python 3."""
    if os.path.exists(directory):
        assert exist_ok, 'Specified directory "{}" already exists.'.format(directory)
    else:
        os.makedirs(directory)
    return


def is_ndds_dataset(input_dir, data_extension="json"):

    # Input argument handling
    # Expand user shortcut if it exists
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )

    data_full_ext = "." + data_extension

    dirlist = os.listdir(input_dir)

    # Find json files
    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]

    # Extract name from json file
    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    is_ndds_dataset = True if data_names else False

    return is_ndds_dataset


def find_ndds_data_in_dir(
    input_dir, data_extension="json", image_extension=None, requested_image_types="all",
):

    # Input argument handling
    # Expand user shortcut if it exists
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir)

    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )
    data_full_ext = "." + data_extension

    if image_extension is None:
        # Auto detect based on list of image extensions to try
        # In case there is a tie, prefer the extensions that are closer to the front
        image_exts_to_try = ["png", "jpg"]
        num_image_exts = []
        for image_ext in image_exts_to_try:
            num_image_exts.append(len([f for f in dirlist if f.endswith(image_ext)]))
        max_num_image_exts = np.max(num_image_exts)
        idx_max = np.where(num_image_exts == max_num_image_exts)[0]
        # If there are multiple indices due to ties, this uses the one closest to the front
        image_extension = image_exts_to_try[idx_max[0]]
        # Mention to user if there are multiple cases to ensure they are aware of the selection
        if len(idx_max) > 1 and max_num_image_exts > 0:
            print(
                'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "{}".'.format(
                    image_extension
                )
            )
    else:
        assert isinstance(
            image_extension, str
        ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
            type(image_extension)
        )
    image_full_ext = "." + image_extension

    assert (
        requested_image_types is None
        or requested_image_types == "all"
        or isinstance(requested_image_types, list)
    ), "Expected \"requested_image_types\" to be None, 'all', or a list of requested_image_types."

    # Read in json files
    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]

    # Sort candidate data files by name
    data_filenames.sort()

    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    # If there are no matching json files -- this is not an NDDS dataset -- return None
    if not data_names:
        return None, None

    data_paths = [os.path.join(input_dir, f) for f in data_filenames if f[0].isdigit()]

    if requested_image_types == "all":
        # Detect based on first entry
        first_entry_name = data_names[0]
        matching_image_names = [
            f
            for f in dirlist
            if f.startswith(first_entry_name) and f.endswith(image_full_ext)
        ]
        find_rgb = (
            True
            if first_entry_name + ".rgb" + image_full_ext in matching_image_names
            else False
        )
        find_depth = (
            True
            if first_entry_name + ".depth" + image_full_ext in matching_image_names
            else False
        )
        find_cs = (
            True
            if first_entry_name + ".cs" + image_full_ext in matching_image_names
            else False
        )
        if len(matching_image_names) > 3:
            print("Image types detected that are not yet implemented in this function.")
    elif requested_image_types:
        # Check based on known data types
        known_image_types = ["rgb", "depth", "cs"]
        for this_image_type in requested_image_types:
            assert (
                this_image_type in known_image_types
            ), 'Image type "{}" not recognized.'.format(this_image_type)

        find_rgb = True if "rgb" in requested_image_types else False
        find_depth = True if "depth" in requested_image_types else False
        find_cs = True if "cs" in requested_image_types else False

    else:
        find_rgb = False
        find_depth = False
        find_cs = False

    dict_of_lists_images = {}
    n_samples = len(data_names)

    if find_rgb:
        rgb_paths = [
            os.path.join(input_dir, f + ".rgb" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                rgb_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(rgb_paths[n])
        dict_of_lists_images["rgb"] = rgb_paths

    if find_depth:
        depth_paths = [
            os.path.join(input_dir, f + ".depth" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                depth_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(depth_paths[n])
        dict_of_lists_images["depth"] = depth_paths

    if find_cs:
        cs_paths = [
            os.path.join(input_dir, f + ".cs" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                cs_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(cs_paths[n])
        dict_of_lists_images["class_segmentation"] = cs_paths

    found_images = [
        dict(zip(dict_of_lists_images, t)) for t in zip(*dict_of_lists_images.values())
    ]

    # Create output dictionaries
    dict_of_lists = {"name": data_names, "data_path": data_paths}

    if find_rgb or find_depth or find_cs:
        dict_of_lists["image_paths"] = found_images

    found_data = [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]

    # Process config files, which are data files that don't have an associated image
    found_configs = {"camera": None, "object": None, "unsorted": []}
    data_filenames_without_images = [f for f in data_filenames if not f[0].isdigit()]

    for data_filename in data_filenames_without_images:
        if data_filename == "_camera_settings" + data_full_ext:
            found_configs["camera"] = os.path.join(input_dir, data_filename)
        elif data_filename == "_object_settings" + data_full_ext:
            found_configs["object"] = os.path.join(input_dir, data_filename)
        else:
            found_configs["unsorted"].append(os.path.join(input_dir, data_filename))

    return found_data, found_configs 

def make_int(output_as_tensor, resolution):
    output_as_tensor_int = deepcopy(output_as_tensor)
    width, height = resolution
    output_as_tensor_int[:, 0] = torch.clamp(output_as_tensor[:, 0], min=0, max=width-1).type(torch.int64)
    output_as_tensor_int[:, 1] = torch.clamp(output_as_tensor[:, 1], min=0, max=height-1).type(torch.int64)
    return output_as_tensor_int
    # the input is torch.int64 with size num_keypoints x 2
    # 

def find_ndds_seq_data_in_dir(
    input_dir, data_extension="json", image_extension=None, requested_image_types="png",is_ct=None
):
    # input_dir = "/mnt/data/Dream_ty/franka_data"
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
    input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir)
    
    found_data = []
    for each_dir in dirlist:
        if each_dir.endswith(".json"):
            continue
        found_data_this_video = []
        output_dir = os.path.join(input_dir, each_dir)
        # output_dir = "/mnt/data/Dream_ty/franka_data/xxxxx"
        image_exts_to_try = ["_color.png", "json"]
        num_image_exts = []
        for image_ext in image_exts_to_try:
            num_image_exts.append(len([f for f in os.listdir(output_dir) \
                                       if f.endswith(image_ext)]))
        min_num_image_exts = np.min(num_image_exts)
        if min_num_image_exts == 0 or min_num_image_exts == 1: 
            continue
        idx_min = np.where(num_image_exts == min_num_image_exts)[0]
        # print("idx_min", idx_min)
        image_extension = image_exts_to_try[idx_min[0]] # here image_extension is png
        if len(idx_min) > 1 and min_num_image_exts > 0:
#            print(
#            'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "{}".'.format(
#                image_extension
#            )
#            )
            pass
        else:
            assert isinstance(
            image_extension, str
            ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
            type(image_extension)
            )
        
        image_full_ext = "color.png" # ".png"
        data_full_ext = "." + "json" # ".json"
        
        # Read in json files
        dir_list = os.listdir(output_dir) 
        png_paths = [f for f in dir_list if f.endswith(image_full_ext)] 
        png_paths.sort()
        
        data_filenames = [f for f in dir_list if f.endswith(data_full_ext)]
        if len(png_paths) != len(data_filenames):
            print('path', output_dir)
        data_filenames.sort()
        data_filenames = data_filenames[:len(png_paths)] 

        
        assert len(png_paths) == len(data_filenames)
        for png, filename in zip(png_paths, data_filenames):
            assert png[:4] == filename[:4]
        
        data_names = [os.path.join(each_dir, os.path.splitext(f)[0][:4]) for f in data_filenames]
        data_paths = [os.path.join(output_dir, f) for f in data_filenames]
        image_paths = [os.path.join(output_dir, f) for f in png_paths]
        
        # print("png_paths", png_paths)
        length = len(png_paths)
        
#        if length != 30: 
#            continue
        
        assert length >= 2
        
        if is_ct:
            if length == 2:
                this_seq = {}
                this_seq['prev_frame_name'] = data_names[0]
                this_seq["prev_frame_img_path"] = image_paths[0]
                this_seq["prev_frame_data_path"] = data_paths[0]
                this_seq["next_frame_name"] = data_names[1]
                this_seq["next_frame_img_path"] = image_paths[1]
                this_seq["next_frame_data_path"] = data_paths[1]
                found_data_this_video.append(this_seq)
            else:
                for i in range(length-1):
                    prev_ind = int(data_names[i].split('/')[-1])
                    next_ind = int(data_names[i+1].split('/')[-1])
                    if next_ind - prev_ind > 1:
                        continue
                    this_seq = {}
                    this_seq['prev_frame_name'] = data_names[i]
                    this_seq["prev_frame_img_path"] = image_paths[i]
                    this_seq["prev_frame_data_path"] = data_paths[i]
                    this_seq["next_frame_name"] = data_names[i+1]
                    this_seq["next_frame_img_path"] = image_paths[i+1]
                    this_seq["next_frame_data_path"] = data_paths[i+1]
                    found_data_this_video.append(this_seq)
        else:
            for i in range(length):
                this_seq = {}
                this_seq['prev_frame_name'] = data_names[i]
                this_seq["prev_frame_img_path"] = image_paths[i]
                this_seq["prev_frame_data_path"] = data_paths[i]
                this_seq["next_frame_name"] = data_names[i]
                this_seq["next_frame_img_path"] = image_paths[i]
                this_seq["next_frame_data_path"] = data_paths[i]
                found_data_this_video.append(this_seq)

        
        found_data = found_data + found_data_this_video
        # print('each dir', each_dir)
        # print(found_data_this_video)
    # print(found_data)
    return found_data 

def load_camera_intrinsics(camera_data_path):

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


def load_image_resolution(camera_data_path):

    # Input argument handling
    assert os.path.exists(
        camera_data_path
    ), 'Expected path "{}" to exist, but it does not.'.format(camera_data_path)

    # Create YAML/json parser
    data_parser = YAML(typ="safe")

    with open(camera_data_path, "r") as f:
        cam_settings_data = data_parser.load(f.read().replace('\t',''))

    image_width = cam_settings_data["camera_settings"][0]["captured_image_size"][
        "width"
    ]
    image_height = cam_settings_data["camera_settings"][0]["captured_image_size"][
        "height"
    ]
    image_resolution = (image_width, image_height)

    return image_resolution

#def load_depth_keypoints(data_path, keypoint_names):
#    parser = YAML(typ="safe")
#    with open(data_path, "r") as f:
#        data = parser.load(f.read().replace('\t', ' '))
#    data = data[0]
#    object_keypoints = data["keypoints"]
#    keypoint_data = {"positions_wrt_cam" : [], "idx" : []}
#     
#    count = 0
#    for idx, kp_name in enumerate(keypoint_names):
#        while object_keypoints[count]["Name"] != kp_name:
#            count += 1
#        assert (object_keypoints[count]["Name"] == kp_name), \
#            "Expected keypoint '{}' to exist in the datafile '{}', but it does not. \
#            Rather, the keypoints are '{}'".format(
#            kp_name, data_path, object_keypoints[count]['Name']
#            )
#        keypoint_data["idx"].append(kp_name)
#        keypoint_data["positions_wrt_cam"].append(object_keypoints[count]["location_wrt_cam"])
#    
#    return keypoint_data
     
    

def load_keypoints(data_path, object_name, keypoint_names):
    assert os.path.exists(
        data_path
    ), 'Expected data_path "{}" to exist, but it does not.'.format(data_path)

    # Set up output structure
    #print(data_path)
    keypoint_data = {"projections": [],"trans":[],"rot_quat":[], 'idx' : [], 'positions' : [], "positions_wrt_cam" : []}

    # Load keypoints for a particular object for now
    parser = YAML(typ="safe")
    with open(data_path, "r") as f:
        data = parser.load(f.read().replace('\t', ' '))

    assert (
        "objects" in data.keys()
    ), 'Expected "objects" key to exist in data file, but it does not.'

    object_names = [o["class"] for o in data["objects"]]
    assert (
        object_name in object_names
    ), 'Requested object_name "{}" does not exist in the data file objects.'.format(
        object_name
    )

    idx_object = object_names.index(object_name)

    object_data = data["objects"][idx_object]
    object_keypoints = object_data["keypoints"]

    ##################################
    if 'location' in object_data and 'quaternion_xyzw' in object_data:
        transformation = np.array(object_data['pose_transform']).T

        RotateY = torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(3, 3)).double()
        rotation_mat = -torch.from_numpy(transformation[:3, :3]) @ RotateY
        rotation_quat = matrix_to_quaternion(rotation_mat)
        keypoint_data['rot_quat'].append(rotation_quat.numpy().tolist())
        keypoint_data['trans'].append(object_data["location"])

        quat_mat = rotation_mat
        trans = torch.tensor(keypoint_data['trans'][-1]).double()
    
    ##################################

    object_keypoint_names = [kp["name"] for kp in object_keypoints]

    # Process in same order as keypoint_names to retain same order
    #print(keypoint_names)
    for kp_name in keypoint_names:
        assert (
            kp_name in object_keypoint_names
        ), "Expected keypoint '{}' to exist in the data file '{}', but it does not.  Rather, the keypoints are '{}'".format(
            kp_name, data_path, object_keypoint_names
        )

        idx_kp = object_keypoint_names.index(kp_name)
        kp_data = object_keypoints[idx_kp]

        if 'location' in object_data and 'quaternion_xyzw' in object_data:

            kp_position = torch.from_numpy(np.array(kp_data['location'])).double()
            kp_position = quat_mat.T @  ((kp_position - trans).view(3, 1))
            kp_position = (kp_position).cpu().tolist()
            keypoint_data['positions'].append(kp_position)

        #
        kp_position_wrt_cam = kp_data['location']
        if "projected_location" in kp_data:
            kp_projection = kp_data["projected_location"]
            keypoint_data["projections"].append(kp_projection)
        
#        if min(kp_projection) < 0:
#            print(data_path)
#            print(kp_name)
#            print(kp_projection)

        keypoint_data["idx"].append(kp_name)
        keypoint_data["positions_wrt_cam"].append(kp_position_wrt_cam)
        # keypoint_data["projections"].append(kp_projection)

    return keypoint_data
    
def load_seq_keypoints(data_path, object_name, keypoint_names, camera_K):
    assert os.path.exists(
        data_path
    ), 'Expected data_path "{}" to exist, but it does not.'.format(data_path)
    
    
    keypoint_data = {"projections" : [],
                     "idx" : [],
                     "positions_wrt_robot" : [],
                     "positions_wrt_cam" : [],
#                     "positions_wrt_cam_42": [],
#                     "positions_wrt_rob_42": [],
                     }
    # Load keypoints for a particular object for now
    parser = YAML(typ="safe")
    with open(data_path, "r") as f:
        data = parser.load(f.read().replace('\t', ' '))
    
    data = data[0]
    assert (object_name == data['ROBOT NAME'])
    object_keypoints = data["keypoints"]
#    object_joints = data["joints_3n_fixed_42"]
#    object_joints = data["keypoints"]
    
    
    Mat = np.array(object_keypoints[0]["R2C Mat"])
    Inv = Mat.T
    Trans = np.array(object_keypoints[0]["location_wrt_cam"])
    
    count = 0
    for idx, kp_name in enumerate(keypoint_names):
        while object_keypoints[count]["Name"] != kp_name:
            count += 1
        assert (object_keypoints[count]["Name"] == kp_name), \
            "Expected keypoint '{}' to exist in the datafile '{}', but it does not. \
            Rather, the keypoints are '{}'".format(
            kp_name, data_path, object_keypoints[count]['Name']
            )
        keypoint_data["idx"].append(kp_name)
        
        
        projection = np.array(object_keypoints[count]["location_wrt_cam"])
        projection = camera_K @ projection
        projection /= projection[2]
        keypoint_data["projections"].append(projection.tolist()[:2])
        # print("projection", projection)
        
        # print("projection", projection)
        # print("raw", object_keypoints[count]["projected_location"])

        keypoint_data["positions_wrt_cam"].append(object_keypoints[count]["location_wrt_cam"])
        
        x3d_wrt_cam = np.array(object_keypoints[count]["location_wrt_cam"])
        x3d_wrt_robot = (Inv @ (x3d_wrt_cam - Trans)).tolist()
        keypoint_data["positions_wrt_robot"].append(x3d_wrt_robot)
    
#    positions_wrt_cam_42 = [i["location_wrt_cam"] for i in object_joints]
#    positions_wrt_rob_42 = np.array(positions_wrt_cam_42)
#    positinos_wrt_rob_42 = (Inv @ (positions_wrt_rob_42 - Trans)).tolist()
#    
#    keypoint_data["positions_wrt_cam_42"] = positions_wrt_cam_42
#    keypoint_data["positions_wrt_rob_42"] = positions_wrt_rob_42
    
    
    return keypoint_data

def load_depth_keypoints(data_path, object_name, keypoint_names, camera_K):
    assert os.path.exists(
        data_path
    ), 'Expected data_path "{}" to exist, but it does not.'.format(data_path)
    
    
    keypoint_data = {"projections" : [],
                     "idx" : [],
                     "positions_wrt_robot" : [],
                     "positions_wrt_cam" : [],
#                     "positions_wrt_cam_42": [],
#                     "positions_wrt_rob_42": [],
                     }
    # Load keypoints for a particular object for now
    parser = YAML(typ="safe")
    with open(data_path, "r") as f:
        data = parser.load(f.read().replace('\t', ' '))
    
    data = data[0]
    assert (object_name == data['ROBOT NAME'])
    object_keypoints = data["keypoints"]
    object_joints = data["joints_3n_fixed_42"]
    
    
    Mat = np.array(object_keypoints[0]["R2C_mat"])
    Inv = Mat.T
    Trans = np.array(object_keypoints[0]["location_wrt_cam"]).reshape(1, 3)
    
    positions_wrt_cam_42 = [i["location_wrt_cam"] for i in object_joints]  
    positions_wrt_rob_42 = np.array(positions_wrt_cam_42)
    #print(positions_wrt_rob_42.shape)
    #print(Trans.shape)
    positions_wrt_rob_42 = ((Inv @ (positions_wrt_rob_42 - Trans).T).T).tolist()
    
    keypoint_data["positions_wrt_cam"] = positions_wrt_cam_42
    keypoint_data["positions_wrt_robot"] = positions_wrt_rob_42
    
    projections = np.array(positions_wrt_cam_42)
    projections = (camera_K @ projections.T).T
    projections[:, :2] /= projections[:, 2:3]
    keypoint_data["projections"] = projections[:, :2].tolist()
    
    
    
    
    return keypoint_data

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
  
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
  
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
  
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_umich_gaussian(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = int(center[0]), int(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        res = [0, 0]  
        # print(res)
        # res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=2, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
    
def draw_umich_gaussian_teaser(heatmap, center, radius, k=1):
    # import pdb; pdb.set_trace()
    diameter = 2 * radius + 1
    height, width = heatmap.shape[0:2]
    x, y = int(center[0]), int(center[1])
    # gaussian = gaussian2D((diameter, diameter), sigma=2)
    
    if x - radius >=0 and x + radius + 1 < width and y - radius >= 0 and y + radius + 1 < height:
        # res = [0, 0]  
        # print(res)
        res = [center[0] -x , center[1] - y]
        gaussian = gaussian2D((diameter, diameter), sigma=6, res=res)
        # height, width = heatmap.shape[0:2]
          
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        # import pdb; pdb.set_trace()
        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
          np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma, res):
    res_x, res_y = res
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-((x-res_x)**2 + (y-res_y)**2) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
    
def _get_aug_param(c, s, width, height, disturb=False):
    aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
    w_border = _get_border(128, width)
    h_border = _get_border(128, height)
    c[0] = np.random.randint(low=w_border, high=width - w_border)
    c[1] = np.random.randint(low=h_border, high=height - h_border)
    
    return c, aug_s

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result 

def _get_input(img, trans_input, input_w, input_h, mean, std):
    inp = cv2.warpAffine(img, trans_input, 
                        (input_w, input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - mean) /std
    inp = inp.transpose(2, 0, 1)
    return inp

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def affine_transforms(pts, t, width, height):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) 
    new_pts = new_pts.T
    

    return new_pts

def affine_transform_and_clip(pts, t, width, height, raw_width, raw_height,mode=None):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) 
    new_pts = new_pts.T
    
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, width-1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, height-1)
    
    new_pts = new_pts.tolist()
    out = []
    
    # flag=  False
    for kp in range(n_kp):
        pts_x, pts_y = pts[kp][0], pts[kp][1]
        if 0.0 <= pts_x < raw_width and 0.0 <= pts_y < raw_height:
            out.append(new_pts[kp])
        else:
            out.append([0,0])
#    if flag:
#        print('pts', pts)
#        print('new_pts', new_pts)
#        print('out', out)
#    if mode:
#        if len(out) != 7:
#            print('n_kp', n_kp)
#            print(len(out))
#            print(out)
    return np.array(out)

def affine_transform_and_clip_old(pts, t, width, height):
    n_kp, _ = pts.shape
    new_ones = np.ones(n_kp).reshape(n_kp, 1)
    new_pts = np.concatenate((pts, new_ones), axis=-1)
    new_pts = np.dot(t, new_pts.T) 
    new_pts = new_pts.T
    
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, width-1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, height-1)
    return new_pts

def get_prev_hm(kp_projs_raw, trans_input, input_w, input_h, raw_width, raw_height, hm_disturb = 0.05, lost_disturb=0.1,mode=None):
    hm_w, hm_h = input_w, input_h
    kp_projs_net_output = affine_transform_and_clip(kp_projs_raw, trans_input, input_w, input_h, raw_width, raw_height,mode=mode) 
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    n_kp, _ = kp_projs_net_output.shape
    radius = 4
    for i in range(n_kp):
        ct = deepcopy(kp_projs_net_output[i])
        ct[0] = ct[0] + np.random.randn() * hm_disturb * 2 # hm_disturb: \lambda_jt
        ct[1] = ct[1] + np.random.randn() * hm_disturb * 2
        
        conf = 1 if np.random.random() > lost_disturb else 0
        draw_umich_gaussian(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn
        
#        if np.random.random() < fp_disturb:
#            # Hard code heatmap disturb ratio, haven't tried other numbers.
#            ct2 = deepcopy(kp_projs_net_output[i])
#            ct2[0] = ct2[0] + np.random.randn() * 0.05 * 2
#            ct2[1] = ct2[1] + np.random.randn() * 0.05 * 2 
#            draw_umich_gaussian(pre_hm, ct2, radius, k=conf)

    return pre_hm 

def get_prev_hm_old(kp_projs_raw, trans_input,input_w, input_h, hm_disturb = 0.05, lost_disturb=0.1):
    hm_w, hm_h = input_w, input_h
    kp_projs_net_output = affine_transform_and_clip_old(kp_projs_raw, trans_input, input_w, input_h) 
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    n_kp, _ = kp_projs_net_output.shape
    radius = 4
    for i in range(n_kp):
        ct = deepcopy(kp_projs_net_output[i])
        ct[0] = ct[0] + np.random.randn() * hm_disturb * 2 # hm_disturb: \lambda_jt
        ct[1] = ct[1] + np.random.randn() * hm_disturb * 2
        
        conf = 1 if np.random.random() > lost_disturb else 0
        draw_umich_gaussian(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn
        
#        if np.random.random() < fp_disturb:
#            # Hard code heatmap disturb ratio, haven't tried other numbers.
#            ct2 = deepcopy(kp_projs_net_output[i])
#            ct2[0] = ct2[0] + np.random.randn() * 0.05 * 2
#            ct2[1] = ct2[1] + np.random.randn() * 0.05 * 2 
#            draw_umich_gaussian(pre_hm, ct2, radius, k=conf)

    return pre_hm 

def get_prev_hm_wo_noise_old(kp_projs_raw, trans_input,input_w, input_h):
    hm_w, hm_h = input_w, input_h
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    if kp_projs_raw is not None:
        kp_projs_net_output = affine_transform_and_clip_old(kp_projs_raw, trans_input, input_w, input_h) 
        n_kp, _ = kp_projs_net_output.shape
        radius = 4
        for i in range(n_kp):
            ct = deepcopy(kp_projs_net_output[i])
            conf = 1
            draw_umich_gaussian(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn

    return pre_hm 

def get_prev_hm_wo_noise(kp_projs_raw, trans_input,input_w, input_h, raw_width, raw_height,mode=None):
    hm_w, hm_h = input_w, input_h
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    if kp_projs_raw is not None:
        kp_projs_net_output = affine_transform_and_clip(kp_projs_raw, trans_input, input_w, input_h, raw_width, raw_height,mode=mode) 
        n_kp, _ = kp_projs_net_output.shape
        radius = 4
        for i in range(n_kp):
            ct = deepcopy(kp_projs_net_output[i])
            conf = 1
            draw_umich_gaussian(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn

    return pre_hm 
    
def get_prev_hm_wo_noise_teaser(kp_projs_raw, trans_input,input_w, input_h, raw_width, raw_height,mode=None):
    hm_w, hm_h = input_w, input_h
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    if kp_projs_raw is not None:
        kp_projs_net_output = affine_transform_and_clip(kp_projs_raw, trans_input, input_w, input_h, raw_width, raw_height,mode=mode)
        n_kp, _ = kp_projs_net_output.shape
        radius = 12
        for i in range(n_kp):
            ct = deepcopy(kp_projs_net_output[i])
            conf = 1
            draw_umich_gaussian_teaser(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn

    return pre_hm 

def get_prev_hm_wo_noise_dream(kp_projs_raw, raw_width, raw_height):
    pre_hm = np.zeros((raw_height, raw_width), dtype=np.float32)
    if kp_projs_raw is not None:
        n_kp, _ = kp_projs_raw.shape
        radius = 12
        for i in range(n_kp):
            ct = deepcopy(kp_projs_raw[i])
            conf = 1
            draw_umich_gaussian_teaser(pre_hm, ct, radius, k=conf)
    
    return pre_hm

def get_prev_hm_wo_noise_cls(kp_projs_raw, kp_gts_raw, trans_input, input_w, input_h, \
                             raw_width, raw_height, mode=None):
    n_kp, _ = kp_gts_raw.shape
    hm_w, hm_h = input_w, input_h
    pre_hm_cls = np.zeros((n_kp, int(hm_h), int(hm_w)), dtype=np.float32)
    if kp_projs_raw is not None:
        assert kp_projs_raw.shape[0] == n_kp
        kp_projs_net_output = affine_transform_and_clip(kp_projs_raw, trans_input, input_w, input_h, raw_width, raw_height,mode=mode) 
        radius = 4
        for i in range(n_kp):
            ct = deepcopy(kp_projs_net_output[i])
            conf = 1
            draw_umich_gaussian(pre_hm_cls[i], ct, radius, k=conf)
    return pre_hm_cls



def get_prev_ori_hm(kp_projs_net_input_np, input_resolution, hm_disturb = 0.05, lost_disturb=0.1, fp_disturb=0.1):
    hm_w, hm_h = input_resolution
    pre_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
    n_kp, _ = kp_projs_net_input_np.shape
    radius = 4
    for i in range(n_kp):
        ct = deepcopy(kp_projs_net_input_np[i])
        ct[0] = ct[0] + np.random.randn() * hm_disturb * 2 # hm_disturb: \lambda_jt
        ct[1] = ct[1] + np.random.randn() * hm_disturb * 2
        
        conf = 1 if np.random.random() > lost_disturb else 0
        draw_umich_gaussian(pre_hm, ct, radius, k=conf) # lost_disturb: fn randomly removing detections with probability \lambda_fn
        
        if np.random.random() < fp_disturb:
            # Hard code heatmap disturb ratio, haven't tried other numbers.
            ct2 = deepcopy(kp_projs_net_input_np[i])
            ct2[0] = ct2[0] + np.random.randn() * 0.05 * 2
            ct2[1] = ct2[1] + np.random.randn() * 0.05 * 2 
            draw_umich_gaussian(pre_hm, ct2, radius, k=conf)

    return pre_hm
        
def get_hm(kp_projs_net_output, output_w, output_h):
    n_kp, _ = kp_projs_net_output.shape
    
    # print('output_h', output_h)
    gt_hm = np.zeros((n_kp, int(output_h), int(output_w)), dtype=np.float32)
    radius = 4
    for i in range(n_kp):
        ct = deepcopy(kp_projs_net_output[i])
        draw_umich_gaussian(gt_hm[i], ct, radius)   
    
    return gt_hm
         


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
