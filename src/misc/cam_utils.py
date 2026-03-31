import cv2
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange,repeat
import traceback
import pytorch3d.transforms as transforms


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


####### Pose update from delta

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(cam_trans_delta: Float[Tensor, "batch 3"],
                cam_rot_delta: Float[Tensor, "batch 3"],
                extrinsics: Float[Tensor, "batch 4 4"],
                # original_rot: Float[Tensor, "batch 3 3"],
                # original_trans: Float[Tensor, "batch 3"],
                # converged_threshold: float = 1e-4
                ):
    # extrinsics is c2w, here we need w2c as input, so we need to invert it
    bs = cam_trans_delta.shape[0]

    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    T_w2c = extrinsics.inverse()

    new_w2c_list = []
    for i in range(bs):
        new_w2c = SE3_exp(tau[i]) @ T_w2c[i]
        new_w2c_list.append(new_w2c)

    new_w2c = torch.stack(new_w2c_list, dim=0)
    return new_w2c.inverse()

    # converged = tau.norm() < converged_threshold
    # camera.update_RT(new_R, new_T)
    #
    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    # return converged


#######  Pose estimation
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')



    
def get_pnp_pose(pts3d, opacity, K, H, W, opacity_threshold=0.3, initial_pose=None):
    pixels = np.mgrid[:W, :H].T.astype(np.float32)
    pts3d = pts3d.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    K = K.clone().detach().cpu().numpy()

    K[0, :] = K[0, :] * W
    K[1, :] = K[1, :] * H

    mask = opacity > opacity_threshold


    try:
        if initial_pose is None:
            res = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                                    iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        else:
            # Extract rotation (3x3) and translation (3x1)
            R = initial_pose[:3, :3]  # Top-left 3x3
            init_tvec = initial_pose[:3, 3].reshape(3, 1)  # Top-right 3x1

            # Convert Rotation Matrix to Rotation Vector
            init_rvec, _ = cv2.Rodrigues(R)

            res = cv2.solvePnPRansac(
                pts3d[mask], 
                pixels[mask], 
                K, 
                None,
                rvec=init_rvec,   # Provide initial rotation
                tvec=init_tvec,   # Provide initial translation
                useExtrinsicGuess=True,  # Use the provided initial pose
                iterationsCount=100, 
                reprojectionError=5, 
                flags=cv2.SOLVEPNP_ITERATIVE  # Use a flag that supports initialization
            )

        
        success, R, T, inliers = res
    except Exception as e:
        print(f"PNP error: {e}")
        traceback.print_exc()
        success = False
    

    if success:
        assert success

        R = cv2.Rodrigues(R)[0]  # world to cam
        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world

        return torch.from_numpy(pose.astype(np.float32))
    else:
        return torch.eye(4).to(torch.float32)


def get_pnp_pose_batch(pts3d, opacity, K, H, W, opacity_threshold=0.3):
    '''
        pts3d: (b, v, h, w, 3)
        opacity: (b, v, h, w)
        K: (b, v, 3, 3)
    '''

    b, v = pts3d.shape[:2]
    device = pts3d.device
    estimated_poses = []
    for i in range(b):
        im_poses = [None]*v

        for j in range(v):
            
            try:
                res = get_pnp_pose(pts3d[i,j], opacity[i,j], K[i,j], H, W, opacity_threshold)
            except Exception as e:
                print(f"PNP error: {e}")
                traceback.print_exc()
                res = None


            if res is not None:
                im_poses[j] = res.to(device)
            else:
                im_poses[j] = torch.eye(4, device=device).to(torch.float32)
            
        poses_v = torch.stack(im_poses, dim=0)


        estimated_poses.append(poses_v)


    c2w_poses = torch.stack(estimated_poses, dim=0) # (b, v, 4, 4)
    return c2w_poses



def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        # aucs.append(np.trapz(r, x=e) / t)
        aucs.append(np.trapezoid(r, x=e) / t)

    
    aucs = [float(x) for x in aucs]
    return aucs



def convert_pose_to_4x4(out):
    B = out.shape[0]
    out_r = out[:,:6]
    out_t = out[:, 6:]
    device = out.device

    out_r = transforms.rotation_6d_to_matrix(out_r)  # [N,3,3]
    pose = torch.zeros((B, 4, 4), device=device)
    pose[:, :3, :3] = out_r
    pose[:, :3, 3] = out_t
    pose[:, 3, 3] = 1.
    return pose


def project_to_cam(pts3d, im_poses, intrinsics, return_z=False):
    '''
        pts3d : [b, n, 3]
        im_poses : cam to world, [b, 4, 4]
        intrinsics : [b, 3, 3]
    '''

    im_poses = torch.inverse(im_poses) # camera_to_world -> world_to_camera
    camera_coords = torch.einsum("bij, bnj -> bni", im_poses[:, :3, :3], pts3d) + im_poses[:, None, :3, 3] # [b, n, 3]

    pred_px_b31 = torch.einsum("bij, bnj -> bni", intrinsics, camera_coords) # [b, n, 3]
    pred_px_b31[..., 2].clamp_(min=1e-6)

    # Dehomogenise.
    pred_px = pred_px_b31[..., :2] / pred_px_b31[..., 2, None] #  [b, n, 2]

    if return_z:
        return pred_px, camera_coords[..., 2, None]
    return pred_px


def depth_projector(pts3d, im_poses):
    '''
        pts3d : [b, n, 3]
        im_poses : cam to world, [b, 4, 4]
    '''
    im_poses = torch.inverse(im_poses)
    camera_coords = torch.einsum("bij, bnj -> bni", im_poses[:, :3, :3], pts3d) + im_poses[:, None, :3, 3] # [b, n, 3]

    return camera_coords[..., 2, None]


def unproject_depth_map_to_point_map_batch(
    depth_map: torch.Tensor,  # (B, H, W) or (B, H, W, 1)
    extrinsics_cam: torch.Tensor,  # (B, 3, 4) or (B, 4, 4)
    intrinsics_cam: torch.Tensor  # (B, 3, 3)
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates in torch.

    Args:
        depth_map: (B, H, W) or (B, H, W, 1)
        extrinsics_cam: (B, 3, 4) or (B, 4, 4)
        intrinsics_cam: (B, 3, 3)

    Returns:
        world_points: (B, H, W, 3)
    """
    if depth_map.ndim == 4:
        depth_map = depth_map.squeeze(-1)  # (B, H, W)

    B, H, W = depth_map.shape

    device = depth_map.device
    dtype = depth_map.dtype

    # Generate pixel grid
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # (W, H)
    grid_u = grid_u / W
    grid_v = grid_v / H
    ones = torch.ones_like(grid_u)
    pixel_coords = torch.stack((grid_u, grid_v, ones), dim=0).float()  # (3, H, W)
    # print("pixel_coords", pixel_coords)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 3, H, W)

    # Flatten pixel coords
    pixel_coords_flat = pixel_coords.view(B, 3, -1)  # (B, 3, H*W)

    # Inverse intrinsics
    intrinsics_inv = torch.inverse(intrinsics_cam)  # (B, 3, 3)

    # Get normalized camera rays
    rays = torch.bmm(intrinsics_inv, pixel_coords_flat)  # (B, 3, H*W)

    # Scale by depth
    depth_flat = depth_map.view(B, 1, -1)  # (B, 1, H*W)
    cam_points_flat = rays * depth_flat  # (B, 3, H*W)

    # Convert to homogeneous coords
    ones = torch.ones((B, 1, cam_points_flat.shape[2]), device=device, dtype=dtype)
    cam_points_hom = torch.cat([cam_points_flat, ones], dim=1)  # (B, 4, H*W)

    # Ensure extrinsic is (B, 4, 4)
    # if extrinsics_cam.shape[1:] == (3, 4):
    #     bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 1, 4).repeat(B, 1, 1)
    #     extrinsics_cam = torch.cat([extrinsics_cam, bottom], dim=1)  # (B, 4, 4)

    # extrinsics_inv = torch.inverse(extrinsics_cam)  # (B, 4, 4)
    # extrinsics_inv = closed_form_inverse_se3(extrinsics_cam)

    extrinsics_inv = extrinsics_cam # should be cam to world

    world_points_flat = torch.bmm(extrinsics_inv, cam_points_hom)  # (B, 4, H*W)
    world_points_flat = world_points_flat[:, :3, :]  # (B, 3, H*W)
    world_points = world_points_flat.transpose(1, 2).view(B, H, W, 3)  # (B, H, W, 3)

    return world_points

def simple_intrin_matrix_from_fov(fov_tensor: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch 3 3"]:
    b = fov_tensor.shape[0]
    intrinsics = torch.eye(3).to(fov_tensor)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()

    fovx, fovy = fov_tensor.chunk(2, dim=-1)
    fx = 0.5 / torch.tan(fovx.squeeze(-1) * 0.5)
    fy = 0.5 / torch.tan(fovy.squeeze(-1) * 0.5)
    cx = cy = torch.ones_like(fx) * 0.5

    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    return intrinsics.float()