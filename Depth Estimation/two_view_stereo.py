import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import torch
import math


from two_view_stereo.dataloader import load_middlebury_data
from two_view_stereo.utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    H_i = K_i_corr @ R_irect @ np.linalg.inv(K_i)
    H_j = K_j_corr @ R_jrect @ np.linalg.inv(K_j)
    
    # Create the destination image size
    size_new = (w_max, h_max)
    
    # Use perspective warping to rectify the images
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, size_new)
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, size_new)
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_iw, R_jw : [3,3]
    T_iw, T_jw : [3,1]
        p_i = R_iw @ p_w + T_iw
        p_j = R_jw @ p_w + T_jw
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ij @ p_j + T_ij, B is the baseline
    """

    """Student Code Starts"""
    R_ij = R_iw @ R_jw.T
    T_ij = T_iw - R_iw @ R_jw.T @ T_jw
    B = float(np.linalg.norm(T_ij))
    
    """Student Code Ends"""

    return R_ij, T_ij, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)

    R_irect=np.zeros((3,3))
    R_irect[0,:]=np.cross(e_i,[0,0,1])
    R_irect[0,:]=R_irect[0,:]/np.linalg.norm(R_irect[0,:])
    R_irect[1,:]=e_i/np.linalg.norm(e_i)
    R_irect[2,:]= np.cross(R_irect[0,:],R_irect[1,:])

    return R_irect

def ct_kernel(src, dst, in_cuda=False):
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    src = torch.from_numpy(src).cuda() if not in_cuda else src
    dst = torch.from_numpy(dst).cuda() if not in_cuda else dst
    K = int(np.sqrt(src.shape[1]))
    src = src.view(src.shape[0], K, K, 3)
    dst = dst.view(dst.shape[0], K, K, 3)

    center = K // 2
    src_center = src[:, center, center, :].unsqueeze(1).unsqueeze(1)
    dst_center = dst[:, center, center, :].unsqueeze(1).unsqueeze(1)
    src_census = (src > src_center).float()
    dst_census = (dst > dst_center).float()

    src_bits = src_census.reshape(src.shape[0], -1)
    dst_bits = dst_census.reshape(dst.shape[0], -1)

    M, N = src_bits.shape[0], dst_bits.shape[0]
    src_exp = src_bits.unsqueeze(1).expand(-1, N, -1)
    dst_exp = dst_bits.unsqueeze(0).expand(M, -1, -1)
    hamming_dist = torch.sum((src_exp != dst_exp).float(), dim=2)

    return hamming_dist.cpu().numpy() if not in_cuda else hamming_dist
        

def ssd_kernel(src, dst, in_cuda=False):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = torch.from_numpy(src).cuda() if not in_cuda else src
    dst = torch.from_numpy(dst).cuda() if not in_cuda else dst
    src_sq = (src ** 2).sum(dim=(1, 2)).unsqueeze(1)
    dst_sq = (dst ** 2).sum(dim=(1, 2)).unsqueeze(0)
    dot = torch.matmul(src.view(src.shape[0], -1), dst.view(dst.shape[0], -1).t())
    ssd = src_sq + dst_sq - 2 * dot
    ssd = ssd.cpu().numpy() if not in_cuda else ssd

    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst, in_cuda=False):
    """Compute SAD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = torch.from_numpy(src).cuda() if not in_cuda else src
    dst = torch.from_numpy(dst).cuda() if not in_cuda else dst
    src_exp = src.unsqueeze(1)
    dst_exp = dst.unsqueeze(0)
    sad = torch.abs(src_exp - dst_exp).sum(dim=(2,3))
    sad = sad.cpu().numpy() if not in_cuda else sad

    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst, in_cuda=False):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = torch.from_numpy(src).cuda() if not in_cuda else src
    dst = torch.from_numpy(dst).cuda() if not in_cuda else dst
    total_zncc = torch.zeros((src.shape[0], dst.shape[0]), device='cuda')
    for c in range(3):
        src_c = src[..., c]
        dst_c = dst[..., c]
        mean_src = src_c.mean(dim=1, keepdim=True)
        mean_dst = dst_c.mean(dim=1, keepdim=True)
        norm_src = src_c - mean_src
        norm_dst = dst_c - mean_dst
        std_src = torch.sqrt(torch.sum(norm_src**2, dim=1, keepdim=True) + 1e-12)
        std_dst = torch.sqrt(torch.sum(norm_dst**2, dim=1, keepdim=True) + 1e-12)
        channel_zncc = torch.matmul(norm_src, norm_dst.t()) / (
            torch.matmul(std_src, std_dst.t()) * src_c.shape[1]
        )
        total_zncc += channel_zncc
    
    zncc = (total_zncc / 3.0)
    zncc = zncc.cpu().numpy() if not in_cuda else zncc

    """Student Code Ends"""

    return zncc * (-1.0)  # M,N

def sobel(m):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
    
    grad_x = torch.nn.functional.conv2d(m, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(m, sobel_y, padding=1)
    mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)
    return mag.squeeze(1)

def sobel_zncc_kernel(src, dst, in_cuda=False):
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]
    src = torch.from_numpy(src).cuda().float() if not in_cuda else src.float()
    dst = torch.from_numpy(dst).cuda().float() if not in_cuda else dst.float()
    patch_size = int(np.sqrt(src.shape[1]))
    total_zncc = torch.zeros((src.shape[0], dst.shape[0]), device='cuda')
    for c in range(3):
        src_c = src[..., c].reshape(-1, patch_size, patch_size)
        dst_c = dst[..., c].reshape(-1, patch_size, patch_size)
        src_c = src_c.unsqueeze(1)
        dst_c = dst_c.unsqueeze(1)
        src_grad = sobel(src_c)
        dst_grad = sobel(dst_c)
        src_grad = src_grad.view(src_grad.size(0), -1)
        dst_grad = dst_grad.view(dst_grad.size(0), -1)
        mean_src = src_grad.mean(dim=1, keepdim=True)
        mean_dst = dst_grad.mean(dim=1, keepdim=True)
        norm_src = src_grad - mean_src
        norm_dst = dst_grad - mean_dst
        std_src = torch.sqrt(torch.sum(norm_src**2, dim=1, keepdim=True) + 1e-12)
        std_dst = torch.sqrt(torch.sum(norm_dst**2, dim=1, keepdim=True) + 1e-12)
        channel_zncc = torch.matmul(norm_src, norm_dst.t()) / (
            torch.matmul(std_src, std_dst.t()) * src_grad.shape[1]
        )
        total_zncc += channel_zncc
    
    zncc = (total_zncc / 3.0)
    zncc = zncc.cpu().numpy() if not in_cuda else zncc

    return zncc * (-1.0)

def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    height, width, color = image.shape
    padding = k_size // 2
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    patch_buffer = np.zeros((height, width, k_size**2, color))

    for i in range(height):
        for j in range(width):
            patch_buffer[i, j] = image[i:i+k_size, j:j+k_size, :].reshape(-1, color)    

    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation 
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    h, w = rgb_i.shape[:2]
    
    disp_map = np.zeros((h, w), dtype=np.float64)
    lr_consistency_mask = np.zeros((h, w), dtype=np.float64)
    
    patches_i = img2patch_func(rgb_i.astype(float) / 255.0, k_size)
    patches_j = img2patch_func(rgb_j.astype(float) / 255.0, k_size)
    
    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vj_idx[None, :] - vi_idx[:, None] + d0
    valid_disp_mask = disp_candidates >= 0.0

    for u in range(w):

        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = kernel_func(buf_i, buf_j)
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper
        
        for v in range(h):
            best_matched_right_pixel = value[v].argmin()
            best_matched_left_pixel = value[:, best_matched_right_pixel].argmin()
            disp_map[v, u] = disp_candidates[v, best_matched_right_pixel]
            lr_consistency_mask[v, u] = float(best_matched_left_pixel == v)
    
    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    dep_map = np.where(disp_map > 0, B * K[0, 0] / disp_map, 0) # depth calculation
    H, W = disp_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]  # focal lengths
    u0, v0 = K[0, 2], K[1, 2]  # principal points

    x_norm = (u - u0) / fx
    y_norm = (v - v0) / fy

    xyz_cam = np.zeros((H, W, 3), dtype=np.float32)
    
    xyz_cam[..., 0] = x_norm * dep_map # 3D X, ... for 3D array
    xyz_cam[..., 1] = y_norm * dep_map 
    xyz_cam[..., 2] = dep_map # z is now equal to depth

    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_cw,
    T_cw,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""

    R_wc = R_cw.T # transpose of rotation matrix
    T_wc = -R_wc @ T_cw.reshape(3, 1) # camera -> world
    pcl_world = (R_wc @ pcl_cam.T).T + T_wc.T # transformation
    
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color



def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_iw,
        T_wc=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
