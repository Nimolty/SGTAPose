import time

import numpy as np
import sympy as sp
import torch
import cv2
import torch.nn.functional as F
from itertools import chain
import ctypes
so = ctypes.cdll.LoadLibrary("/DATA/disk1/hyperplane/robot_pose_3090/DREAM-master/dream_geo/rf_tools/libtestso_final.so")
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
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
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
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


def compute_rotation_matric_from_quaternion(quaternion):
    quaternion_norm = quaternion / np.linalg.norm(quaternion)
    qw, qx, qy, qz = quaternion_norm
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    R = np.array([[1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw],
                  [2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw],
                  [2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy]])
    return R
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w_new, x_new, y_new, z_new])

def get_new_point_from_quaternion(pt_input, quat):
    pt_input = np.insert(pt_input, 0, 0.)
    quat_conjugate = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    pt_new = quaternion_multiply(quaternion_multiply(quat, pt_input), quat_conjugate)
    return np.array([pt_new[1], pt_new[2], pt_new[3]])


# 功能：计算残差
# 输入：当前更新变量的值value，测量点pc_input，目标点pc_target，全局符号变量global_symbols，每个点对应的权重weights
# 输出：当前value计算得到的残差
def fun(value,x2d_input,x3d_input, weights,camera):
    [[fx, _, cx], [_, fy, cy], [_, _, _]] = camera
    f_dev = []
    qw = value[0]
    qx = value[1]
    qy = value[2]
    qz = value[3]
    tx = value[4]
    ty = value[5]
    tz = value[6]
    for i in range(len(x2d_input)):
        x3d_tmp = x3d_input[i]
        x2d_tmp = x2d_input[i]
        weight_tmp = weights[i]
        x3d = x3d_tmp[0]
        y3d = x3d_tmp[1]
        z3d = x3d_tmp[2]
        x2d = x2d_tmp[0]
        y2d = x2d_tmp[1]
        wx2d = weight_tmp[0]
        wy2d = weight_tmp[1]

        f_dev_i = wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))**2
        f_dev.append(f_dev_i)
        f_dev_i = wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))**2
        f_dev.append(f_dev_i)
    f_dev_constrain = 1e8*(qw**2 + qx**2 + qy**2 + qz**2 - 1)**2 + 1e8*(qw**2 + qx**2 + qy**2 + qz**2 - 1)**2
    f_dev.append(f_dev_constrain)
    f_dev = np.array(f_dev)

    return f_dev

# 功能：获得雅可比矩阵
# 输入：当前更新变量的值value，测量点pc_input，目标点pc_target，全局符号变量global_symbols，每个点对应的权重weights
# 输出：当前value计算得到的雅可比矩阵
def dfun(value,x2d_input,x3d_input,weights,camera):
    [[fx, _, cx], [_, fy, cy], [_, _, _]] = camera
    df = []
    qw = value[0]
    qx = value[1]
    qy = value[2]
    qz = value[3]
    tx = value[4]
    ty = value[5]
    tz = value[6]
    for i in range(len(x2d_input)):
        x3d_tmp = x3d_input[i]
        x2d_tmp = x2d_input[i]
        weight_tmp = weights[i]
        x3d = x3d_tmp[0]
        y3d = x3d_tmp[1]
        z3d = x3d_tmp[2]
        x2d = x2d_tmp[0]
        y2d = x2d_tmp[1]
        wx2d = weight_tmp[0]
        wy2d = weight_tmp[1]

        df_0 = []
        df_0.append(wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cx*(2*qw*z3d + 2*qx*y3d - 2*qy*x3d) + fx*(2*qw*x3d + 2*qy*z3d - 2*qz*y3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))*(-2*qw*z3d - 2*qx*y3d + 2*qy*x3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_0.append(wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cx*(2*qw*y3d - 2*qx*z3d + 2*qz*x3d) + fx*(2*qx*x3d + 2*qy*y3d + 2*qz*z3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))*(-2*qw*y3d + 2*qx*z3d - 2*qz*x3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_0.append(wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cx*(-2*qw*x3d - 2*qy*z3d + 2*qz*y3d) + fx*(2*qw*z3d + 2*qx*y3d - 2*qy*x3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))*(2*qw*x3d + 2*qy*z3d - 2*qz*y3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_0.append(wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cx*(2*qx*x3d + 2*qy*y3d + 2*qz*z3d) + fx*(-2*qw*y3d + 2*qx*z3d - 2*qz*x3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))*(-2*qx*x3d - 2*qy*y3d - 2*qz*z3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_0.append(-2*fx*wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))
        df_0.append(0)
        df_0.append(wx2d**2*(x2d - (cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*cx/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + 2*(cx*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fx*(qw*(qw*x3d + qy*z3d - qz*y3d) - qx*(-qx*x3d - qy*y3d - qz*z3d) + qy*(qw*z3d + qx*y3d - qy*x3d) - qz*(qw*y3d - qx*z3d + qz*x3d) + tx))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_1 = []
        df_1.append(wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cy*(2*qw*z3d + 2*qx*y3d - 2*qy*x3d) + fy*(2*qw*y3d - 2*qx*z3d + 2*qz*x3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))*(-2*qw*z3d - 2*qx*y3d + 2*qy*x3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_1.append(wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cy*(2*qw*y3d - 2*qx*z3d + 2*qz*x3d) + fy*(-2*qw*z3d - 2*qx*y3d + 2*qy*x3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))*(-2*qw*y3d + 2*qx*z3d - 2*qz*x3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_1.append(wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cy*(-2*qw*x3d - 2*qy*z3d + 2*qz*y3d) + fy*(2*qx*x3d + 2*qy*y3d + 2*qz*z3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))*(2*qw*x3d + 2*qy*z3d - 2*qz*y3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_1.append(wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*(cy*(-2*qw*x3d - 2*qy*z3d + 2*qz*y3d) + fy*(2*qx*x3d + 2*qy*y3d + 2*qz*z3d))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) - 2*(cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))*(2*qw*x3d + 2*qy*z3d - 2*qz*y3d)/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df_1.append(0)
        df_1.append(-2*fy*wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))
        df_1.append(wy2d**2*(y2d - (cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz))*(-2*cy/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + 2*(cy*(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz) + fy*(qw*(qw*y3d - qx*z3d + qz*x3d) - qx*(qw*z3d + qx*y3d - qy*x3d) - qy*(-qx*x3d - qy*y3d - qz*z3d) + qz*(qw*x3d + qy*z3d - qz*y3d) + ty))/(qw*(qw*z3d + qx*y3d - qy*x3d) + qx*(qw*y3d - qx*z3d + qz*x3d) - qy*(qw*x3d + qy*z3d - qz*y3d) - qz*(-qx*x3d - qy*y3d - qz*z3d) + tz)**2))
        df.append(df_0)
        df.append(df_1)
    df_constrain = []
    qw = value[0]
    qx = value[1]
    qy = value[2]
    qz = value[3]
    wx2d = 1e8
    wy2d = 1e8
    df_constrain.append(4*qw*wx2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1) + 4*qw*wy2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1))
    df_constrain.append(4*qx*wx2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1) + 4*qx*wy2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1))
    df_constrain.append(4*qy*wx2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1) + 4*qy*wy2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1))
    df_constrain.append(4*qz*wx2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1) + 4*qz*wy2d*(qw**2 + qx**2 + qy**2 + qz**2 - 1))
    df_constrain.append(0)
    df_constrain.append(0)
    df_constrain.append(0)
    df.append(df_constrain)
    df = np.array(df)
    return df


def GN(value,x2d_input,x3d_input, weights,camera):

    i = 0
    delta = np.ones((7,1))*100
    while np.sum(abs(delta)) > 1e-4 and i < 200:
        # 控制循环次数
        df = dfun(value,x2d_input,x3d_input,weights,camera)
        f = fun(value,x2d_input,x3d_input, weights,camera)
        value1 = value - np.dot(np.dot(np.linalg.inv(np.array(np.dot(df.T, df), dtype='float32') + 1e-4 * np.identity(df.shape[1])), df.T), f)
        delta = value1 - value          # 比较x的变化
        value = value1
        i += 1
    return value


# 功能：利用高斯牛顿实现加权迭代对齐
# 输入：当前更新变量的值value，测量点pc_input，目标点pc_target，全局符号变量global_symbols，每个点对应的权重weights
# 输出：测量点到目标点的四元数以及平移矩阵
def register_GN(x2d_input,x3d_input,quat_init,T_init,weights,camera):

    value_init = np.hstack((quat_init, T_init))[0]
    # (value_init, x2d_input, x3d_input, weights, camera) = (np.array([-6.00401775e-01,  6.94723604e-01,  3.64914466e-01, -1.53994337e-01,
    #     4.27258903e+02,  4.77330930e+02,  5.10517971e+02]), [[-23764.255859375, -28227.865234375], [9696.9287109375, 9325.146484375], [3059.188232421875, 3003.78857421875], [-2022.2001953125, -2832.12255859375], [12641.9052734375, 14991.693359375], [-6165.609375, -8699.5908203125], [5291.4384765625, 4608.4716796875]], [[403.0658264160156, 513.8037719726562, 550.1163330078125], [575.7579345703125, 502.5988464355469, 518.3341064453125], [365.4688720703125, 379.2112731933594, 474.35394287109375], [401.3437805175781, 620.1107788085938, 583.8678588867188], [484.8558044433594, 479.5791015625, 597.5640869140625], [250.0081787109375, 505.1006774902344, 541.8153076171875], [527.6553344726562, 464.54168701171875, 449.271240234375]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [100000000.0, 100000000.0]], np.array([[615.5,   0. , 615.5],
    #    [  0. , 320. , 240. ],
    #    [  0. ,   0. ,   1. ]], dtype=float))

    # print((value_init, x2d_input, x3d_input, weights, camera))
    # print(value_init, x2d_input, x3d_input, weights, camera,sep = '\n')
    s = time.perf_counter()
    value = GN(value_init,x2d_input,x3d_input, weights,camera)
    ss = time.perf_counter()
    print(ss-s)
    return value[:4], value[4:]

def register_GN_C(x2d_input_p,x3d_input_p,quat_init,T_init,weights,camera, num_points):
    # quat_init(1,4,xyzw),T_init(1,3),x2d_input( num_points,2),x3d_input( num_points,3),weights( num_points+2,2),camera(3,3))
    value_init_l = (ctypes.c_double * 7)(quat_init[0, 0], quat_init[0, 1], quat_init[0, 2], quat_init[0, 3], T_init[0, 0], T_init[0, 1],T_init[0, 2])
    x2d_input = (ctypes.c_double * (num_points * 2))(*(list(chain.from_iterable(x2d_input_p))))
    x3d_input = (ctypes.c_double * (num_points * 3))(*(list(chain.from_iterable(x3d_input_p))))
    weightl = (ctypes.c_double * (num_points * 2 + 2))(*(list(chain.from_iterable(weights))))
    cameral = (ctypes.c_double * 9)(*list(chain.from_iterable(camera.tolist())))
    ans = (ctypes.c_double * 7)(*([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    so.LM(value_init_l, x2d_input, x3d_input, weightl, cameral, ans, num_points)
    listans = np.array(list(ans))
    return listans[:4], listans[4:]

def quaternion_normlize(quat):
    norm = (np.dot(quat, quat.T)) ** 0.5
    return quat/(norm + 1e-8)
# 功能：设置点权重
# 输入：点的数量num_pt
# 输出：点权重weights，其中weights[-1]为单位四元数约束的权重，其余权重顺序与点输入顺序一致
def get_weights_without(num_pt):
    weights = np.ones((num_pt+1,2), dtype=float)
    weights[-1:] = 1e8   # 四元数约束项
    return weights.tolist()

def get_weights(num_pt,distance):
    weights = np.ones((num_pt+1,2), dtype=float)
    for j in range(2):
        for i in range(num_pt):
            # if distance[i,j] > 100:
            #     weights[i,j] = 0
            # # elif distance[i,j] < 1:
            # #     weights[i,j] = 1
            # else:
            #     weights[i,j] = np.power(1000,(1-(distance[i,j]/100)))/1000

            # some other choice
            # if distance[i, j] > 1:
            #     weights[i,j] = 0
            # else:
            #     weights[i, j] = np.power(1, (1 - distance[i, j]/1)) * 1

            # e ^ (-x)
            # weights[i,j] = np.exp(- distance[i, j])

            # e ^ (-2x)
            # weights[i,j] = np.exp(- 2 * distance[i, j])

            # e ^ (-5x)
            # weights[i, j] = np.exp(-5 * distance[i, j])

            # e ^ (-5.5x)
            weights[i, j] = np.exp(-5 * distance[i, j])

            # e ^ (-8x)
            # weights[i, j] = np.exp(-8 * distance[i, j])

            # e^(-15x)
            # weights[i, j] = np.exp(- 15 * distance[i, j])

    # distance[:, 0] /= (max(distance[:,0]))
    # distance[:, 1] /= (max(distance[:, 1]))
    # weights[0:-1] -= distance
    # print(distance)
    weights[-1:] = 1e8   # 四元数约束项
    # print(weights)
    return weights.tolist()

def get_weights_real(x2d_input,x3d_input,transform,camera):
    num_points,_ = x2d_input.shape
    weights = np.zeros((num_points+1,2))

    for i in range(num_points):
        x2d_tmp = x2d_input[i]
        if(x2d_tmp[0] < -1000):
            continue
        x3,y3,z3 = x3d_input[i]
        x3d_tmp = np.array([x3,y3,z3,1])
        x2d_rep = camera @ transform[0:3] @ x3d_tmp
        x2d_rep[0]/=x2d_rep[2]
        x2d_rep[1]/=x2d_rep[2]
        x2d_rep = x2d_rep[0:2]
        dis = (x2d_rep - x2d_tmp)**2
        for j in range(2):
            if dis[j] > 100:
                weights[i,j] = 0
            elif dis[j] < 1:
                weights[i,j] = 1
            else:
                weights[i,j] = np.power(1000,(1-(dis[j]/10)))/1000
        
    weights[-1] = [1e8,1e8]
    return weights,num_points

def make_one_pose(n_points,camera_intrinsic):
    translation_gt = ((torch.randn([1,3])+5) * 100).unsqueeze(2)
    rotation_quaternion_gt = torch.randn(1,4)
    rotation_quaternion_gt = F.normalize(rotation_quaternion_gt, dim=-1)  # normalize to unit quaternion9
    rotation_matrix_gt = quaternion_to_matrix(rotation_quaternion_gt)
    pose_extrinsics = torch.cat((rotation_matrix_gt,translation_gt),dim = -1)

    x3d = (torch.randn([n_points,3]) + 5) * 100
    x3d_h = torch.cat(((x3d, torch.ones(n_points,1))),dim = -1).unsqueeze(2)
    x2d_h = torch.matmul(torch.from_numpy(camera_intrinsic),torch.matmul(pose_extrinsics, x3d_h))
    for i in range(n_points):
        x2d_h[i, 0] /= x2d_h[i, 2]
        x2d_h[i, 1] /= x2d_h[i, 2]
        x2d_h[i, 2] /= x2d_h[i, 2]
    x2d = x2d_h[:,:2].squeeze()
    # x2d[4] += 2
    x3d_np = np.array(x3d, dtype=np.float64)
    x2d_np = np.array(x2d, dtype=np.float64)
    # print(x2d_np.shape,x3d_np.shape)
    distCoeffs = np.asarray([0, 0, 0, 0, 0], dtype=np.float64)
    _, rval, tval, inliers = cv2.solvePnPRansac(x3d_np, x2d_np, camera_intrinsic, distCoeffs,5.0, cv2.SOLVEPNP_P3P)
    rmat, _ = cv2.Rodrigues(rval)
    rqua = matrix_to_quaternion(torch.from_numpy(rmat).unsqueeze(0))

    return np.array(translation_gt.squeeze(2)).tolist(),np.array(rotation_quaternion_gt).tolist(),np.array(x2d).tolist(),np.array(x3d).tolist(), np.array(rqua), np.array(tval).T

if __name__ == '__main__':
    x3d,y3d,z3d = sp.symbols('x3d,y3d,z3d')
    x2d,y2d = sp.symbols('x2d,y2d')
    wx2d,wy2d = sp.symbols('wx2d,wy2d')
    w = sp.symbols('w')         # 配准权重
    qw, qx, qy, qz = sp.symbols('qw, qx, qy, qz')   # 四元数
    tx, ty, tz = sp.symbols('tx, ty, tz')           # 平移矩阵
    camera = np.asarray([[615.5, 0, 615.5], [0, 320, 240], [0, 0, 1]], dtype=np.float32)

    global_symbols = [x3d,x2d,y3d,y2d,wx2d,wy2d,z3d,
                      qw, qx, qy, qz, tx, ty, tz]

    translation_gt, rotation_quaternion_gt, x2d_7_2, x3d_7_3, rqua,tval = make_one_pose(7,camera)
    # print(get_new_point_from_quaternion(x3d_7_3[0], rqua[0]))
    x2d_rep = []
    ind = 0
    for x in np.array(x3d_7_3):
        x2d_rep_i = camera @ (get_new_point_from_quaternion(x, rqua[0]) + tval).T
        x2d_rep_i[0] /= x2d_rep_i[2]
        x2d_rep_i[1] /= x2d_rep_i[2]
        x2d_rep.append(x2d_rep_i[0:2])


    weights = get_weights_without(7)


    sta = time.perf_counter()
    quat, T = register_GN(x2d_7_2,x3d_7_3,rqua,tval,weights,camera)
    mid = time.perf_counter()
    ans = register_GN_C(x2d_7_2,x3d_7_3,rqua,tval,weights,camera,7)
    end = time.perf_counter()
    print("c_time:",end-mid,"python_time:",mid - sta,sep = '\n')
    # print(quat.shape,T.shape)
    quat = quaternion_normlize(quat)
    print("GT",rotation_quaternion_gt,translation_gt)
    print("削减权重",quat,T)
    print("init:",rqua,tval)
    print("C",ans)
